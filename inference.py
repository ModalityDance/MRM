import os, json, argparse, logging
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import bt_loss, build_prism_dataset, build_reddit_dataset, log_and_print


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


def evaluate_maml(model, val_loader, args, split_name: str):
    device = args.device
    total_loss, train_loss = 0.0, 0.0
    task_count = 0

    user_stats = defaultdict(lambda: [0, 0])  # correct, total

    for batch in val_loader:
        if isinstance(batch, list):
            batch = batch[0]

        support_ch = batch["train_chosen"].to(device).float().squeeze(0)
        support_rj = batch["train_rejected"].to(device).float().squeeze(0)
        val_ch     = batch["val_chosen"].to(device).float().squeeze(0)
        val_rj     = batch["val_rejected"].to(device).float().squeeze(0)

        # same as your training-time eval: support = train + val, query = test
        support_ch = torch.cat([support_ch, val_ch], dim=0)
        support_rj = torch.cat([support_rj, val_rj], dim=0)
        query_ch   = batch["test_chosen"].to(device).float().squeeze(0)
        query_rj   = batch["test_rejected"].to(device).float().squeeze(0)
        user_id    = batch["user_id"][0]

        # per-user adaptation: update only shared_weight
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
        train_loss += (sum(loss_sup_sum) / len(loss_sup_sum)) if loss_sup_sum else 0.0
        task_count += 1

    user_accs = [
        correct / total if total > 0 else 0.0
        for correct, total in user_stats.values()
    ]
    assert len(user_accs) == len(val_loader), f"Expected {len(val_loader)} user accuracies, got {len(user_accs)}"

    avg_loss = total_loss / task_count if task_count > 0 else float("inf")
    avg_loss_sup = train_loss / task_count if task_count > 0 else float("inf")
    return user_accs, avg_loss, avg_loss_sup


def load_ckpt_into_model(model: nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location="cpu")

    # common ckpt formats: raw state_dict OR wrapped dict
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state_dict = state["model"]
    else:
        state_dict = state

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return missing, unexpected


def parse_args():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--embed_pt", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--dataset", type=str, default="PRISM", choices=["PRISM", "REDDIT"])
    ap.add_argument("--val_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42, help="Only used for dataset splitting")

    # dataset options (keep consistent with training)
    ap.add_argument("--seen_train_limit", type=int, default=-1)
    ap.add_argument("--unseen_train_limit", type=int, default=-1)
    ap.add_argument("--data_augmentation", action="store_true")
    ap.add_argument("--score_threshold", type=float, default=-1)

    # model
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hidden_layers", type=int, nargs="*", default=[2048, 1024, 512, 256])
    ap.add_argument("--use_bias", action="store_true")
    ap.add_argument("--input_dim", type=int, default=4096)

    # eval-time adaptation hyperparams (same meaning as training-time eval)
    ap.add_argument("--inner_lr", type=float, default=1e-3)
    ap.add_argument("--eval_inner_epochs", type=int, default=5)

    # system
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return ap.parse_args()


def mean_std(arr, ddof=0):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=ddof))


if __name__ == "__main__":
    args = parse_args()

    emb = torch.load(args.embed_pt, map_location="cpu", weights_only=True)
    with open(args.meta_json, "r") as f:
        meta = json.load(f)

    print(f"-----------Raw data stats-------------")
    print(f"Total users: {len(meta)}")
    print(f"Total pairs: {sum(len(entries) for entries in meta.values())}")
    print(f"Embedding shape: {emb.shape}")

    if args.dataset == "PRISM":
        seen_dataset, unseen_dataset = build_prism_dataset(
            meta, emb,
            seen_train_limit=args.seen_train_limit,
            unseen_train_limit=args.unseen_train_limit,
            seed=args.seed,
            val_ratio=args.val_ratio,
            aug=args.data_augmentation,
            threshold=args.score_threshold,
        )
    else:
        seen_dataset, unseen_dataset = build_reddit_dataset(
            meta, emb,
            seen_train_limit=args.seen_train_limit,
            unseen_train_limit=args.unseen_train_limit,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )

    seen_loader = DataLoader(seen_dataset, batch_size=1, shuffle=False)
    unseen_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False)

    # build + load ckpt
    model = MRM(args.input_dim, args.hidden_layers, use_bias=args.use_bias)
    missing, unexpected = load_ckpt_into_model(model, args.ckpt, args.device)
    print(f"Model loaded from {args.ckpt}")
    if missing or unexpected:
        msg = f"load_state_dict: missing={missing}, unexpected={unexpected}"
        print(msg)

    # inference (with per-user adaptation)
    seen_accs, seen_loss, seen_sup_loss = evaluate_maml(model, seen_loader, args, "seen")
    unseen_accs, unseen_loss, unseen_sup_loss = evaluate_maml(model, unseen_loader, args, "unseen")

    seen_acc = float(np.mean(seen_accs)) if len(seen_accs) > 0 else float("nan")
    unseen_acc = float(np.mean(unseen_accs)) if len(unseen_accs) > 0 else float("nan")
    overall_acc = float(np.mean(seen_accs + unseen_accs)) if (len(seen_accs) + len(unseen_accs)) > 0 else float("nan")

    print(f"Seen loss {seen_loss:.4f}, Seen support loss {seen_sup_loss:.4f}")
    print(f"Unseen loss {unseen_loss:.4f}, Unseen support loss {unseen_sup_loss:.4f}")
    print(f"Seed={args.seed} | Seen acc {seen_acc:.3f} | Unseen acc {unseen_acc:.3f} | Overall {overall_acc:.3f}")


