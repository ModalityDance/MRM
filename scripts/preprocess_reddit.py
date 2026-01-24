from datasets import load_dataset
from datasets import load_from_disk
from collections import defaultdict
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import pandas as pd
import pickle
import random
import json
import torch
import sys
import torch.nn.functional as F
from typing import List, Dict

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

df_train = pd.DataFrame(dataset["train"])
df_val = pd.DataFrame(dataset["validation"])

worker_results_train = {}
worker_results_val = {}


for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing train"):
    worker_id = row['worker']
    text = row['info']['post']
    summaries = row['summaries']
    choice = row['choice']

    
    winning = summaries[choice]['text']
    losing = summaries[1 - choice]['text']
    
    if worker_id not in worker_results_train:
        worker_results_train[worker_id] = []
    worker_results_train[worker_id].append({
        'text': text,
        'winning_summary': winning,
        'losing_summary': losing
    })


for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Processing validation"):
    worker_id = row['worker']
    text = row['info']['post']
    summaries = row['summaries']
    choice = row['choice']
    
    
    winning = summaries[choice]['text']
    losing = summaries[1 - choice]['text']
    
    if worker_id not in worker_results_val:
        worker_results_val[worker_id] = []
    worker_results_val[worker_id].append({
        'text': text,
        'winning_summary': winning,
        'losing_summary': losing
    })

all_worker_results_train = dict(sorted(worker_results_train.items(), key=lambda item: len(item[1]), reverse=False))
all_worker_results_val = dict(sorted(worker_results_val.items(), key=lambda item: len(item[1]), reverse=False))

train_workers_set = set(all_worker_results_train.keys())
val_workers_set = set(all_worker_results_val.keys())
common_workers = list(train_workers_set & val_workers_set)

print(f"Number of total workers: {len(common_workers)}")

common_worker_data = []

for worker_id in common_workers:
    if worker_id in all_worker_results_train:
        for entry in all_worker_results_train[worker_id]:
            common_worker_data.append({
                "user_id": worker_id,
                "prompt": entry["text"],
                "chosen": entry["winning_summary"],
                "rejected": entry["losing_summary"],
                "split": "train",
            })
    if worker_id in all_worker_results_val:
        for entry in all_worker_results_val[worker_id]:
            common_worker_data.append({
                "user_id": worker_id,
                "prompt": entry["text"],
                "chosen": entry["winning_summary"],
                "rejected": entry["losing_summary"],
                "split": "val",
            })

print(f"Total samples: {len(common_worker_data)}")

@torch.no_grad()
def extract_embeddings_single_sample(model, head, tokenizer, worker_data, save_prefix="emb_output"):
    model.eval()
    all_data = worker_data

    user_to_samples: Dict[str, List[Dict]] = {}
    for sample in all_data:  
        user_id = sample["user_id"]
        user_to_samples.setdefault(user_id, []).append(sample)

    all_embeddings = []
    metadata_per_user = {}
    current_idx = 0
    conv_id = 0

    for user_id, samples in tqdm(user_to_samples.items(), desc="Processing users"):
        metadata_per_user[user_id] = []
        for sample in samples:

            conv_chosen = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["chosen"]}
            ]
            conv_rejected = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["rejected"]}
            ]

            pos_text = tokenizer.apply_chat_template(conv_chosen, tokenize=True, return_tensors="pt").to(model.device)
            neg_text = tokenizer.apply_chat_template(conv_rejected, tokenize=True, return_tensors="pt").to(model.device)

            outputs_pos = model(pos_text, output_hidden_states=True)
            hidden_pos = outputs_pos.hidden_states[-1]  # [1, T, H]
            pos_score = outputs_pos.logits[0][0].item()
            last_emb_pos = hidden_pos[0, -1].to("cpu")  # [H]
            cls_score = head(last_emb_pos.unsqueeze(0).to(model.device)).item()
            if abs(cls_score - pos_score) > 1e-4:
                print("CLS score mismatch:", cls_score, pos_score)
            all_embeddings.append(last_emb_pos)

            chosen_idx = current_idx
            current_idx += 1

            outputs_neg = model(neg_text, output_hidden_states=True)
            hidden_neg = outputs_neg.hidden_states[-1]  # [1, T, H]
            neg_score = outputs_neg.logits[0][0].item()
            last_emb_neg = hidden_neg[0, -1].to("cpu")  # [H]
            cls_score = head(last_emb_neg.unsqueeze(0).to(model.device)).item()
            if abs(cls_score - neg_score) > 1e-4:
                print("CLS score mismatch:", cls_score, neg_score)
            all_embeddings.append(last_emb_neg)

            rejected_idx = current_idx
            current_idx += 1

            conv_id += 1
            
            metadata_per_user[user_id].append({
                "conversation_id": conv_id,
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "chosen_score": pos_score,
                "rejected_score": neg_score,
                "chosen_idx": chosen_idx,
                "rejected_idx": rejected_idx,
                "split": sample["split"],
            })

    print(all_embeddings[0].shape)
    all_embeddings_tensor = torch.stack(all_embeddings, dim=0)
    print(f"Total embeddings shape: {all_embeddings_tensor.shape}")
    torch.save(all_embeddings_tensor, f"{save_prefix}.pt")

    with open(f"{save_prefix}.json", "w") as f:
        json.dump(metadata_per_user, f, indent=2)

    print(f"✅ Saved {len(all_embeddings)} embeddings to {save_prefix}.pt")
    print(f"✅ Metadata saved to {save_prefix}.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the reward model"
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        required=True,
        help="Prefix path to save extracted embeddings"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch_dtype = torch.bfloat16
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="flash_attention_2",
    )

    cls_head = model.score

    extract_embeddings_single_sample(
        model,
        cls_head,
        tokenizer,
        common_worker_data,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()
