
import json, os, random, argparse
from typing import List, Dict
import torch, torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import sys
import torch
from collections import defaultdict, Counter, OrderedDict
import torch.nn.functional as F
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup, AutoModelForSequenceClassification
from higher import innerloop_ctx
import math
import matplotlib.pyplot as plt
from copy import deepcopy

from utils import bt_loss, evaluate_train, build_prism_dataset, log_and_print, build_reddit_dataset


class MRM(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        hidden_sizes: List[int] = [32],
        use_bias: bool = True,
        scale: float = 0.01,
    ):
        super().__init__()
        self.scale = scale

        self.layers = nn.ModuleList()
        last = in_dim
        for h in hidden_sizes:
            self.layers.append(nn.Linear(last, h, bias=use_bias))
            last = h

        self.shared_weight = nn.Parameter(torch.randn(last))

    @staticmethod
    def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
        return torch.softmax(w, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x * self.scale

        for layer in self.layers:
            x = layer(x)  

        w = self._normalize_weights(self.shared_weight)      
        reward = (x * w).sum(dim=-1)                          
        return reward.view(-1)


def evaluate_maml(model, val_loader, args, type):
    device = args.device
    total_loss, train_loss = 0.0, 0.0
    task_count = 0

    user_stats = defaultdict(lambda: [0, 0])  
    user_accs = []

    for batch in val_loader:
        if isinstance(batch, list):
            batch = batch[0]

        support_ch = batch["train_chosen"].to(device).float().squeeze(0)
        support_rj = batch["train_rejected"].to(device).float().squeeze(0)
        val_ch   = batch["val_chosen"].to(device).float().squeeze(0)
        val_rj   = batch["val_rejected"].to(device).float().squeeze(0)

        support_ch = torch.cat([support_ch, val_ch], dim=0)
        support_rj = torch.cat([support_rj, val_rj], dim=0)
        query_ch   = batch["test_chosen"].to(device).float().squeeze(0)
        query_rj   = batch["test_rejected"].to(device).float().squeeze(0)
        user_id    = batch["user_id"][0]

        fast_model = deepcopy(model).to(device)
        fast_model.train()

        updated_param = fast_model.shared_weight
        inner_opt = torch.optim.Adam([updated_param], lr=args.inner_lr)

        loss_sup_sum = []
        for _ in range(args.eval_inner_epochs):
            inner_opt.zero_grad()
            s_ch = fast_model(support_ch)
            s_rj = fast_model(support_rj)
            loss_sup = bt_loss(s_ch, s_rj)
            loss_sup.backward()
            inner_opt.step()
            loss_sup_sum.append(loss_sup.item())

        fast_model.eval()
        with torch.no_grad():
            score_ch = fast_model(query_ch)
            score_rj = fast_model(query_rj)
            loss_q = bt_loss(score_ch, score_rj)

            correct = (score_ch > score_rj).sum().item()
            total = score_ch.size(0)

        user_stats[user_id][0] += correct
        user_stats[user_id][1] += total

        total_loss += loss_q.item()
        train_loss += sum(loss_sup_sum) / len(loss_sup_sum) if loss_sup_sum else 0.0
        task_count += 1

    user_accs = [
        correct / total if total > 0 else 0.0
        for correct, total in user_stats.values()
    ]
    assert len(user_accs) == len(val_loader), f"Expected {len(val_loader)} user accuracies, got {len(user_accs)}" 

    avg_loss = total_loss / task_count if task_count > 0 else float("inf")
    avg_loss_sup = train_loss / task_count if task_count > 0 else float("inf")

    return user_accs, avg_loss, avg_loss_sup


def RPO(
    losses: torch.Tensor,
    tail_frac: float,
    gamma: float = 0.0,
) -> torch.Tensor:
    assert 0.0 < tail_frac <= 1.0 
    L = losses.flatten()

    tau = torch.quantile(L, q=1.0 - tail_frac)

    if gamma is None or gamma <= 0.0:
        mask = (L > tau).float().detach()
        obj = (mask * L).sum()
        return obj

    w = torch.sigmoid((L - tau) / gamma)
    obj = (w * L).sum()

    return obj


def maml_train(args, seen_dataset):
    device = args.device
    model = MRM(args.input_dim, args.hidden_layers, use_bias=args.use_bias).to(device)
            
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    total_steps = len(seen_dataset) * args.epochs // args.tasks_per_batch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=meta_opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_acc = 0.0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        seen_dataset.epoch = epoch
        train_loader = DataLoader(seen_dataset, batch_size=1, shuffle=True)
        model.train()
        epoch_loss = 0.0

        loss_buf = []          

        meta_opt.zero_grad()

        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]

            support_ch = batch["train_chosen"].to(device).float().squeeze(0)   # [S, D]
            support_rj = batch["train_rejected"].to(device).float().squeeze(0) # [S, D]
            query_ch   = batch["val_chosen"].to(device).float().squeeze(0)     # [Q, D]
            query_rj   = batch["val_rejected"].to(device).float().squeeze(0)   # [Q, D]
            user_id    = batch["user_id"][0]

            updated_param = model.shared_weight
            inner_opt = torch.optim.Adam([updated_param], lr=args.inner_lr)

            with innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(args.train_inner_epochs):
                    s_ch = fmodel(support_ch)
                    s_rj = fmodel(support_rj)
                    loss_sup = bt_loss(s_ch, s_rj)
                    diffopt.step(loss_sup)

                q_ch = fmodel(query_ch)
                q_rj = fmodel(query_rj)
                query_loss = bt_loss(q_ch, q_rj)

            
            loss_buf.append(query_loss)
            epoch_loss += float(query_loss.detach())

            tail = args.rpo_ratio
            gamma = args.rpo_gamma
            if len(loss_buf) == args.tasks_per_batch:
                losses = torch.stack(loss_buf)  

                obj = RPO(losses, tail_frac=tail, gamma=gamma)

                meta_opt.zero_grad()
                obj.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                meta_opt.step()
                scheduler.step()

                loss_buf.clear()
                meta_opt.zero_grad()

        if len(loss_buf) > 0:
            losses = torch.stack(loss_buf)

            obj = RPO(losses, tail_frac=tail, gamma=gamma)

            meta_opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            meta_opt.step()
            scheduler.step()

            loss_buf.clear()
            meta_opt.zero_grad()


        if epoch % args.log_freq == 0:
            avg_loss = epoch_loss / len(train_loader)
            sup_acc, qry_acc = evaluate_train(model, train_loader, device)
            seen_acc_res, seen_loss, seen_train_loss = evaluate_maml(model, seen_val_loader, args, 'seen')
            unseen_acc_res, unseen_loss, unseen_train_loss = evaluate_maml(model, unseen_val_loader, args, 'unseen')

            seen_acc = np.mean(seen_acc_res)
            unseen_acc = np.mean(unseen_acc_res)
            total_acc = np.mean(seen_acc_res + unseen_acc_res)

            logger.info(
                f"[Epoch {epoch}] "
                f"Meta loss: {avg_loss:.4f} | "
                f"Train Sup acc: {sup_acc:.3f} | "
                f"Train Qry acc: {qry_acc:.3f} | "
                f"Eval support loss: {seen_train_loss:.4f} (Seen), {unseen_train_loss:.4f} (Unseen) | "
                f"Eval loss: {seen_loss:.4f} (Seen), {unseen_loss:.4f} (Unseen) | "
                f"Seen acc: {seen_acc:.3f} | "
                f"Unseen acc: {unseen_acc:.3f} | "
                f"Overall acc: {total_acc:.3f} "
            )
            print(f"\nEpoch {epoch} | Loss={avg_loss:.4f}, TrainSup={sup_acc:.3f}, TrainQry={qry_acc:.3f}, Seenloss={seen_loss:.4f}, Uneenloss={unseen_loss:.4f}, 🔶 Seen={seen_acc:.3f}, Unseen={unseen_acc:.3f}, Overall={total_acc:.3f}")

            if args.log_to_wandb:
                wandb.log(
                    {
                        "meta_loss": avg_loss,
                        "train_sup_acc": sup_acc,
                        "train_qry_acc": qry_acc,
                        "seen_acc": seen_acc,
                        "unseen_acc": unseen_acc,
                        "overall_acc": total_acc,
                        "eval_seen_loss": seen_loss,
                        "eval_unseen_loss": unseen_loss,
                        "eval_support_loss": (seen_train_loss + unseen_train_loss) / 2,
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=epoch,    
                )


            if total_acc > best_acc:
                best_acc = total_acc
                best_acc_seen_acc = seen_acc
                best_acc_unseen_acc = unseen_acc
                best_acc_total_acc = total_acc
                logger.info(f"🔶 New best overall acc {best_acc:.4f} at epoch {epoch} (Seen {seen_acc:.4f}, Unseen {unseen_acc:.4f})")
            
        if epoch % args.save_freq == 0:
            save_path = os.path.join(output_dir_name, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    return best_acc_seen_acc, best_acc_unseen_acc, best_acc_total_acc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_pt", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--hidden_layers", type=int, nargs="*", default=[2048, 1024, 512, 256])
    ap.add_argument("--use_bias", action="store_true")
    ap.add_argument("--inner_lr", type=float, default=1e-3)
    ap.add_argument("--meta_lr",  type=float, default=5e-4)
    ap.add_argument("--train_inner_epochs", type=int, default=1)
    ap.add_argument("--eval_inner_epochs", type=int, default=5)
    ap.add_argument("--tasks_per_batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_path", default="output")
    ap.add_argument("--val_ratio", type=float, default=0.8)
    ap.add_argument("--input_dim", type=int, default=4096)
    ap.add_argument("--log_freq", type=int, default=100)
    ap.add_argument("--log_to_wandb", action="store_true")
    ap.add_argument("--save_freq", type=int, default=100)
    ap.add_argument("--eval_at_beginning", action="store_true")
    ap.add_argument("--dataset", type=str, default="PRISM", choices=["PRISM", "REDDIT"])
    ap.add_argument("--seen_train_limit", type=int, default=-1,
                    help="Limit the number of training pairs per seen user, -1 means no limit")
    ap.add_argument("--unseen_train_limit", type=int, default=-1,
                    help="Limit the number of training pairs per unseen user, -1 means no limit")
    ap.add_argument("--data_augmentation", action="store_true",
                    help="Whether to use data augmentation for unseen users")
    ap.add_argument("--score_threshold", type=float, default=-1,
                    help="Threshold for filtering pairs based on score, only used if data_augmentation is True")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Number of times to repeat the training process with different random seeds")
    ap.add_argument("--rpo_ratio", type=float, default=0.5,
                    help="The ratio of users to consider in the tail for RPO evaluation")
    ap.add_argument("--rpo_gamma", type=float, default=0.5,
                    help="Smoothing parameter for RPO, larger values lead to smoother optimization")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seen_acc_rec = []
    unseen_acc_rec = []
    total_acc_rec = []

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"{current_time}_{args.meta_json.split('/')[-1]}_ep{args.epochs}_olr{args.meta_lr}_ilr{args.inner_lr}_bch{args.tasks_per_batch}_Tep{args.train_inner_epochs}_Eep{args.eval_inner_epochs}_vr{args.val_ratio}"
    output_dir_name = os.path.join(args.output_path, output_dir)
    os.makedirs(output_dir_name, exist_ok=True)

    log_file = os.path.join(output_dir_name, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Training arguments: {vars(args)}")
    logger.info(f"All results will be written under {output_dir_name}")
    print(f"All results will be written under {output_dir_name}")

    emb = torch.load(args.embed_pt, map_location="cpu", weights_only=True)
    with open(args.meta_json, "r") as f: meta = json.load(f)

    print(f"-----------Raw data stats-------------")
    print(f"Total users: {len(meta)}")
    print(f"Total pairs: {sum(len(entries) for entries in meta.values())}")
    print(f"Embedding shape: {emb.shape}")


    for rep in range(args.repeat):
        
        current_seed = args.seed + rep
        torch.manual_seed(current_seed)
        random.seed(current_seed)
        np.random.seed(current_seed)
        print(f"\n======= 🔽 Repeat {rep+1}/{args.repeat}, seed={current_seed} =======")

        if args.log_to_wandb:
            wandb.init(
                project="MUM",
                name=output_dir,
                config=vars(args)
            )

        if args.dataset == "PRISM":
            seen_dataset, unseen_dataset = build_prism_dataset(
                meta, emb,
                seen_train_limit=args.seen_train_limit,
                unseen_train_limit=args.unseen_train_limit,
                seed=current_seed,
                val_ratio=args.val_ratio,
                aug=args.data_augmentation,
                threshold=args.score_threshold,
            )
        elif args.dataset == "REDDIT":
            seen_dataset, unseen_dataset = build_reddit_dataset(
                meta, emb,
                seen_train_limit=args.seen_train_limit,
                unseen_train_limit=args.unseen_train_limit,
                seed=current_seed,
                val_ratio=args.val_ratio,
            )

        logger.info(f"Training arguments for repeat {rep+1}:")
        for k, v in vars(args).items():
            logger.info(f"{k}: {v}")
        logger.info(f"Repeat: {rep+1}, seed: {current_seed}")

        num_seen_users = len(seen_dataset)
        num_unseen_users = len(unseen_dataset)

        seen_train_lens = [ex['train_chosen'].shape[0] for ex in seen_dataset]
        seen_val_lens   = [ex['val_chosen'].shape[0] for ex in seen_dataset]
        seen_test_lens  = [ex['test_chosen'].shape[0]  for ex in seen_dataset]
        total_seen_train = int(np.sum(seen_train_lens))
        total_seen_val   = int(np.sum(seen_val_lens))
        total_seen_test  = int(np.sum(seen_test_lens))
        avg_seen_train   = float(np.mean(seen_train_lens))
        avg_seen_val     = float(np.mean(seen_val_lens))
        avg_seen_test    = float(np.mean(seen_test_lens))

        unseen_train_lens = [ex['train_chosen'].shape[0] for ex in unseen_dataset]
        unseen_val_lens   = [ex['val_chosen'].shape[0] for ex in unseen_dataset]
        unseen_test_lens  = [ex['test_chosen'].shape[0]  for ex in unseen_dataset]
        total_unseen_train = int(np.sum(unseen_train_lens))
        total_unseen_val   = int(np.sum(unseen_val_lens))
        total_unseen_test  = int(np.sum(unseen_test_lens))
        avg_unseen_train   = float(np.mean(unseen_train_lens))
        avg_unseen_val     = float(np.mean(unseen_val_lens))
        avg_unseen_test    = float(np.mean(unseen_test_lens))


        print(f"---------Dataset stats-------------")
        log_and_print(logger, f"Seen users   : {num_seen_users}")
        log_and_print(logger, f"Seen train   : {total_seen_train} (avg/user: {avg_seen_train:.1f}) | Seen val : {total_seen_val} (avg/user: {avg_seen_val:.1f}) | Seen test : {total_seen_test} (avg/user: {avg_seen_test:.1f})")
        log_and_print(logger, f"Unseen users : {num_unseen_users}")
        log_and_print(logger, f"Unseen train : {total_unseen_train} (avg/user: {avg_unseen_train:.1f}) | Unseen val : {total_unseen_val} (avg/user: {avg_unseen_val:.1f}) | Unseen test : {total_unseen_test} (avg/user: {avg_unseen_test:.1f})")

        seen_val_loader = DataLoader(seen_dataset, batch_size=1, shuffle=False)
        unseen_val_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False)


        best_acc_seen_acc, best_acc_unseen_acc, best_acc_total_acc = maml_train(args, seen_dataset)

        seen_acc_rec.append(best_acc_seen_acc)
        unseen_acc_rec.append(best_acc_unseen_acc)
        total_acc_rec.append(best_acc_total_acc)

        log_and_print(logger, f"Best Acc  — Seen {best_acc_seen_acc:.3f}, Unseen {best_acc_unseen_acc:.3f}, Overall {best_acc_total_acc:.3f}")


    def mean_std(arr, ddof=0):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr, ddof=ddof))  

    log_and_print(logger, "\n========== Run name: " + output_dir_name + " ==========")
    log_and_print(logger, f"\n========== Final results after {args.repeat} repeats ==========")

    mu, sd = mean_std(seen_acc_rec);   log_and_print(logger, f"Seen acc:       {mu:.3f} ± {sd:.3f}")
    mu, sd = mean_std(unseen_acc_rec); log_and_print(logger, f"Unseen acc:     {mu:.3f} ± {sd:.3f}")
    mu, sd = mean_std(total_acc_rec);  log_and_print(logger, f"Overall acc:    {mu:.3f} ± {sd:.3f}")