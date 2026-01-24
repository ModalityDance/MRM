
import random
import numpy as np
from tqdm import tqdm
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random, torch
from collections import defaultdict, Counter, OrderedDict


def bt_loss(score_pos, score_neg):
    return -nn.functional.logsigmoid(score_pos - score_neg).mean()


@torch.no_grad()
def evaluate_train(model, data_loader, device):
    model.eval()
    
    user_sup_stats = defaultdict(lambda: [0, 0])  
    user_qry_stats = defaultdict(lambda: [0, 0])

    for batch in data_loader:
        if isinstance(batch, list):
            batch = batch[0]

        support_ch = batch["train_chosen"].to(device).float().squeeze(0)
        support_rj = batch["train_rejected"].to(device).float().squeeze(0)
        query_ch   = batch["val_chosen"].to(device).float().squeeze(0)
        query_rj   = batch["val_rejected"].to(device).float().squeeze(0)
        user_id    = batch["user_id"][0]

        score_ch   = model(support_ch)
        score_rj   = model(support_rj)
        score_q_ch = model(query_ch)
        score_q_rj = model(query_rj)

        sup_correct = (score_ch > score_rj).sum().item()
        qry_correct = (score_q_ch > score_q_rj).sum().item()

        user_sup_stats[user_id][0] += sup_correct
        user_sup_stats[user_id][1] += score_ch.size(0)

        user_qry_stats[user_id][0] += qry_correct
        user_qry_stats[user_id][1] += score_q_ch.size(0)

    user_sup_accs = [
        correct / total if total > 0 else 0.0
        for correct, total in user_sup_stats.values()
    ]
    user_qry_accs = [
        correct / total if total > 0 else 0.0
        for correct, total in user_qry_stats.values()
    ]

    sup_acc = sum(user_sup_accs) / len(user_sup_accs) if user_sup_accs else 0.0
    qry_acc = sum(user_qry_accs) / len(user_qry_accs) if user_qry_accs else 0.0

    return sup_acc, qry_acc


def log_and_print(logger, msg):
    print(msg)
    logger.info(msg)


def build_prism_dataset(
    meta,
    emb,
    seen_train_limit=-1,
    unseen_train_limit=-1,
    seed=42,
    val_ratio=0.2,
    aug=True,
    threshold=0.0,
    include_user_ids=None,
):
    if include_user_ids is not None:
        meta = {uid: meta[uid] for uid in include_user_ids if uid in meta}
    user_ids = sorted(meta.keys())
    random.Random(seed).shuffle(user_ids)
    n = len(user_ids)
    seen_user_ids = user_ids[: n // 2]
    unseen_user_ids = user_ids[n // 2 :]

    def make_user_data(user_ids, train_limit):
        user_data = {}
        for uid in user_ids:
            entries = meta[uid]
            conv2entries = defaultdict(list)
            for e in entries:
                conv_id = e["conversation_id"]
                conv2entries[conv_id].append(e)
            conv_ids = list(conv2entries.keys())

            rnd = random.Random(seed + int(hashlib.md5(uid.encode()).hexdigest(), 16) % (2**32)) 
            rnd.shuffle(conv_ids)
            split_pt = len(conv_ids) // 2
            train_ids = conv_ids[:split_pt]
            test_ids  = conv_ids[split_pt:]
            if train_limit > 0:
                train_ids = train_ids[:train_limit]

            def augment_conv(conv_entries):
                prompt2cand = defaultdict(list)
                for e in conv_entries:
                    prompt2cand[str(e["prompt"])].append({
                        "idx": e["chosen_idx"],
                        "score": e["chosen_score"],
                        "emb": emb[e["chosen_idx"]].clone()
                    })
                    prompt2cand[str(e["prompt"])].append({
                        "idx": e["rejected_idx"],
                        "score": e["rejected_score"]-1,
                        "emb": emb[e["rejected_idx"]].clone()
                    })

                pairs = []
                for prompt_id, cand in prompt2cand.items():
                    idx2info = {c["idx"]: c for c in cand}
                    cand_list = list(idx2info.values())
                    n = len(cand_list)
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                            diff = cand_list[i]["score"] - cand_list[j]["score"]
                            if diff > threshold:
                                pairs.append({
                                    "conv_id": conv_entries[0]["conversation_id"],
                                    "prompt": prompt_id,
                                    "chosen_emb": cand_list[i]["emb"],
                                    "rejected_emb": cand_list[j]["emb"],
                                    "chosen_score": cand_list[i]["score"],
                                    "rejected_score": cand_list[j]["score"]-1,
                                })
                return pairs

            train_examples = []
            for cid in train_ids:
                conv_entries = conv2entries[cid]
                if aug:
                    train_examples.extend(augment_conv(conv_entries))
                else:
                    for e in conv_entries:
                        diff = e["chosen_score"] - e["rejected_score"]
                        if diff > threshold:
                            train_examples.append({
                                "conv_id": e["conversation_id"],
                                "prompt": str(e["prompt"]),
                                "chosen_emb": emb[e["chosen_idx"]].clone(),
                                "rejected_emb": emb[e["rejected_idx"]].clone(),
                                "chosen_score": e["chosen_score"],
                                "rejected_score": e["rejected_score"],
                            })

            test_examples = []
            for cid in test_ids:
                for e in conv2entries[cid]:
                    test_examples.append({
                        "conv_id": e["conversation_id"],
                        "prompt": str(e["prompt"]),
                        "chosen_emb": emb[e["chosen_idx"]].clone(),
                        "rejected_emb": emb[e["rejected_idx"]].clone(),
                        "chosen_score": e["chosen_score"],
                        "rejected_score": e["rejected_score"],
                    })
            if len(train_examples) == 0 or len(test_examples) == 0:
                continue 
            user_data[uid] = {
                "train": train_examples,
                "test": test_examples,
            }
        return user_data

    seen_user_data = make_user_data(seen_user_ids, seen_train_limit)
    unseen_user_data = make_user_data(unseen_user_ids, unseen_train_limit)

    seen_dataset = MetaDataset(seen_user_data, val_ratio=val_ratio, seed=seed)
    unseen_dataset = MetaDataset(unseen_user_data, val_ratio=val_ratio, seed=seed)

    return seen_dataset, unseen_dataset


def build_reddit_dataset(
    meta,
    emb,
    val_ratio=0.2,
    seed=42,
    seen_train_limit=-1,
    unseen_train_limit=-1,
    include_user_ids=None,
):
    if include_user_ids is not None:
        meta = {uid: meta[uid] for uid in include_user_ids if uid in meta}
        
    user_ids = sorted(meta.keys())
    random.Random(seed).shuffle(user_ids)
    n = len(user_ids)
    seen_user_ids = user_ids[: n // 2]
    unseen_user_ids = user_ids[n // 2 :]

    def make_user_data(user_ids, train_limit):
        user_data = {}
        for uid in user_ids:
            entry_list = meta[uid]
            train_examples, test_examples = [], []
            pair_idx = 0
            for e in entry_list:
                conv_id = f"conv_{pair_idx}"
                ex = {
                    "conv_id": conv_id,
                    "prompt": conv_id,  
                    "chosen_emb": emb[e["chosen_idx"]].clone(),
                    "rejected_emb": emb[e["rejected_idx"]].clone(),
                    "chosen_score": 100,
                    "rejected_score": 1,
                }
                pair_idx += 1
                if e.get("split") == "train":
                    train_examples.append(ex)
                elif e.get("split") == "val":
                    test_examples.append(ex)
                else:
                    if random.random() < 0.5:
                        train_examples.append(ex)
                    else:
                        test_examples.append(ex)
            rnd = random.Random(seed + int(hashlib.md5(uid.encode()).hexdigest(), 16) % (2**32)) 
            rnd.shuffle(train_examples)        
            if train_limit > 0:
                train_examples = train_examples[:train_limit]
            user_data[uid] = {
                "train": train_examples,
                "test": test_examples,
            }
        return user_data

    seen_user_data = make_user_data(seen_user_ids, seen_train_limit)
    unseen_user_data = make_user_data(unseen_user_ids, unseen_train_limit)

    seen_dataset = MetaDataset(seen_user_data, val_ratio=val_ratio, seed=seed)
    unseen_dataset = MetaDataset(unseen_user_data, val_ratio=val_ratio, seed=seed)

    return seen_dataset, unseen_dataset


class MetaDataset(Dataset):
    def __init__(self, user_data, val_ratio=0.2, seed=42, epoch=0):
        self.user_data = user_data
        self.val_ratio = val_ratio
        self.seed = seed
        self.user_ids = [uid for uid in user_data]
        self.epoch = epoch

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        uid = self.user_ids[index]
        train_examples = self.user_data[uid]["train"]
        test_examples  = self.user_data[uid]["test"]

        conv2examples = defaultdict(list)
        for ex in train_examples:
            conv2examples[ex["conv_id"]].append(ex)
        conv_ids = list(conv2examples.keys())

        rng = random.Random(self.seed + index + 1000 * self.epoch)
        rng.shuffle(conv_ids)

        n_val = max(1, int(len(conv_ids) * self.val_ratio))
        if len(conv_ids) == 1:
            val_ids = sup_ids = set(conv_ids)
        else:
            val_ids = set(conv_ids[:n_val])
            sup_ids = set(conv_ids[n_val:])


        train = [ex for cid in sup_ids for ex in conv2examples[cid]]
        val     = [ex for cid in val_ids for ex in conv2examples[cid]]
        test   = test_examples

        if len(train) == 0:
            raise ValueError(f"User {uid} has empty train.")
        elif len(val) == 0:
            raise ValueError(f"User {uid} has empty val.")
        elif len(test) == 0:
            raise ValueError(f"User {uid} has empty test.")


        def emb_pair(examples):
            ch = torch.stack([ex["chosen_emb"] for ex in examples])
            rj = torch.stack([ex["rejected_emb"] for ex in examples])
            return ch, rj

        train_ch, train_rj = emb_pair(train)
        val_ch, val_rj = emb_pair(val)
        test_ch, test_rj = emb_pair(test)

        return {
            "user_id": uid,
            "train_chosen": train_ch,
            "train_rejected": train_rj,
            "val_chosen": val_ch,
            "val_rejected": val_rj,
            "test_chosen": test_ch,
            "test_rejected": test_rj,
        }

