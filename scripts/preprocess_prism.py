import json
from collections import defaultdict
import os
import torch
from tqdm import tqdm
import sys
from copy import deepcopy
import statistics
import requests
import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs("data/prism", exist_ok=True)
os.makedirs("data/emb/prism", exist_ok=True)

def is_valid_jsonl(filepath):
    """Check if a file is valid JSONL (not HTML rate limit page)"""
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            # Check if it's HTML (rate limit page) or empty
            if first_line.startswith('<!DOCTYPE html>') or first_line.startswith('<html') or not first_line:
                return False
            # Try to parse as JSON to ensure it's valid JSONL
            import json
            json.loads(first_line)
            return True
    except (json.JSONDecodeError, Exception):
        return False

def download_file_with_retry(url, filepath, max_retries=10, delay=30):
    """Download a file with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {filepath} (attempt {attempt + 1})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Validate the downloaded file
            if is_valid_jsonl(filepath):
                print(f"Successfully downloaded and validated {filepath}")
                return True
            else:
                print(f"Downloaded file {filepath} appears to be rate limited or invalid")
                if attempt < max_retries - 1:
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Failed to download valid file after {max_retries} attempts")
                    return False
                    
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"Failed to download after {max_retries} attempts")
                return False
    
    return False

# Download files with retry logic
files_to_download = [
    ("https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/survey.jsonl", "data/prism/survey.jsonl"),
    ("https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/conversations.jsonl", "data/prism/conversations.jsonl"),
    ("https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/utterances.jsonl", "data/prism/utterances.jsonl")
]

for url, filepath in files_to_download:
    if not is_valid_jsonl(filepath):
        print(f"File {filepath} doesn't exist or is invalid, downloading...")
        success = download_file_with_retry(url, filepath)
        if not success:
            print(f"CRITICAL: Failed to download {filepath}. Exiting.")
            sys.exit(1)
    else:
        print(f"File {filepath} already exists and is valid.")
seed=123

import os
import json
import numpy as np
np.random.seed(seed=123)

from pydantic import BaseModel
from typing import List, Optional, Dict

class Demographics(BaseModel):
    self_description: str
    preference: List[str] = []
    age: str
    gender: str
    education: str
    employment: str
    marital: str
    english_proficiency: str

class UserInfo(BaseModel):
    user_id: str
    dialog_ids: List[str] = []
    demographics: Demographics
    system_string: str

class DataUser(BaseModel):
    data: Dict[str, UserInfo] = {}

class Turn(BaseModel):
    turn_nb: int
    user_utterance: List[str] = []
    chosen_utterance: List[str] = []
    rejected_utterance: List[str] = []
    chosen_utterance_scores: List[float] = []
    rejected_utterance_scores: List[float] = []

class DialogInfo(BaseModel):
    dialog_id: str
    user_id: str
    turns: List[Optional[Turn]] = []
    total_turn_nb: int
    open_feedback: str = ""

class DataDialog(BaseModel):
    data: Dict[str, DialogInfo] = {}


data_user = DataUser()

with open ("data/prism/survey.jsonl", 'r') as f:
    for line in f:
        d = json.loads(line)
        if d["num_completed_conversations"] == 0:
            continue
        data_user.data[d["user_id"]] = UserInfo(
            user_id = d["user_id"],
            demographics =  Demographics(
                self_description = d["self_description"],
                preference = [k for k, v in d["order_stated_prefs"].items() if v in [1,2,3]],
                age = d["age"],
                gender = d["gender"],
                education = d["education"],
                employment = d["employment_status"],
                marital = d["marital_status"],
                english_proficiency = d["english_proficiency"]
            ),
            system_string = d["system_string"]
        )

data_dialog = DataDialog()

with open ("data/prism/conversations.jsonl", 'r') as f:
    for line in f:
        d = json.loads(line)
        data_user.data[d["user_id"]].dialog_ids.append(d["conversation_id"])
        data_dialog.data[d["conversation_id"]] = DialogInfo(
            dialog_id = d["conversation_id"],
            user_id = d["user_id"],
            total_turn_nb = d["conversation_turns"],
            turns = [None for _ in range(d["conversation_turns"])],
            open_feedback = d["open_feedback"]
        )
        for utterance in d["conversation_history"]:
            if data_dialog.data[d["conversation_id"]].turns[utterance["turn"]] is None:
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]] = Turn(
                    turn_nb = utterance["turn"]
                )
            if utterance["role"] == "user":
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].user_utterance.append(utterance["content"])
                # data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].chosen_utterance_scores.append(utterance['score'])
            elif utterance["if_chosen"]:
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].chosen_utterance.append(utterance["content"])
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].chosen_utterance_scores.append(utterance['score'])
            else:
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].rejected_utterance.append(utterance["content"])
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].rejected_utterance_scores.append(utterance['score'])

data_dialog = data_dialog.dict()["data"]
data_user = data_user.dict()["data"]

# filter out users with no qualified example
dialog_ids = list(data_dialog.keys())
for dialog_id in dialog_ids:
    qualified_num = data_dialog[dialog_id]["total_turn_nb"]
    for turn in data_dialog[dialog_id]["turns"]:
        if (turn['user_utterance'] == [] 
            or turn['chosen_utterance'] == [] 
            or turn['rejected_utterance'] == [] 
        ):
            qualified_num -= 1
    # only delete when the whole dialogue is not qualifed
    if qualified_num == 0:
        print("delete dialogue", dialog_id, "by", data_dialog[dialog_id]["user_id"])
        data_user[data_dialog[dialog_id]["user_id"]]["dialog_ids"].remove(dialog_id)
        if data_user[data_dialog[dialog_id]["user_id"]]["dialog_ids"] == []:
            print("delete user", data_dialog[dialog_id]["user_id"])
            del data_user[data_dialog[dialog_id]["user_id"]]
        del data_dialog[dialog_id]


import numpy as np
np.random.seed(seed=123)

user_ids = np.array(list(data_user.keys()))
stats = []
for user_id in user_ids:
    stats.append(len(np.array(data_user[user_id]["dialog_ids"])))

print("Total users:", len(user_ids))
print("Avg no. of dialogs: ", np.mean(stats))
print("Std of dialogs: ", np.std(stats))
print("Max no. of dialogs: ", np.max(stats))
print("Min no. of dialogs: ", np.min(stats))

# Previous code following LoRe to select users and dialogs
# Then we will extract embeddings of each pair, textual demographic information and dialog history are optional

min_dialog_threshold = 6
user_ids_to_remove = [uid for uid, uinfo in data_user.items() if len(uinfo["dialog_ids"]) < min_dialog_threshold]
for uid in user_ids_to_remove:
    del data_user[uid]
print(f"Removed {len(user_ids_to_remove)} users with less than {min_dialog_threshold} dialogs.")
print(f"Remaining users: {len(data_user)}")
print(f"Remaining dialogs: {sum(len(uinfo['dialog_ids']) for uinfo in data_user.values())}")
user_ids = np.array(list(data_user.keys()))

add_textual_info = False
add_history = True

aggregate_rejected = False   

output_records = []

for user_id in user_ids:
    for dialog_id in data_user[user_id]["dialog_ids"]:
        dialog = data_dialog[dialog_id]
        history = []

        textual_info = ""
        if add_textual_info:
            demographics = data_user[user_id].get("demographics", {})
            preference = ", ".join(demographics.get("preference", []))
            textual_info = f"User textual information: preference: {preference}; "
            for k, v in demographics.items():
                if k != "preference":
                    textual_info += f"{k}: {v}; "

        for turn in dialog["turns"]:
            if not (turn['user_utterance'] and turn['chosen_utterance'] and turn['rejected_utterance']):
                continue

            conv = []
            if add_history:
                for past in history:
                    conv.append({"role": "user", "content": past["user"]})
                    conv.append({"role": "assistant", "content": past["assistant"]})

            user_msg = turn["user_utterance"][0]
            if add_textual_info and textual_info:
                user_msg = textual_info + "\n" + user_msg
            conv.append({"role": "user", "content": user_msg})

            chosen_txt   = turn["chosen_utterance"][0]
            rejected_list = list(turn["rejected_utterance"])  
            chosen_score  = turn.get("chosen_utterance_scores", [None])[0]
            rejected_scores_full = turn.get("rejected_utterance_scores", [])

            if aggregate_rejected:
                rejected_text = '\n'.join(rejected_list)
                rej_scores = []
                for idx, rej in enumerate(rejected_list):
                    if idx < len(rejected_scores_full):
                        rej_scores.append(rejected_scores_full[idx])
                rejected_score_mean = statistics.mean(rej_scores) if rej_scores else None

                output_records.append({
                    "user_id": user_id,
                    "conversation_id": dialog_id,
                    "conv": deepcopy(conv),
                    "chosen": chosen_txt,
                    "rejected": rejected_text,                
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score_mean,    
                })
            else:
                for idx, rejected in enumerate(rejected_list):
                    rej_score = None
                    if idx < len(rejected_scores_full):
                        rej_score = rejected_scores_full[idx]

                    output_records.append({
                        "user_id": user_id,
                        "conversation_id": dialog_id,
                        "conv": deepcopy(conv),
                        "chosen": chosen_txt,
                        "rejected": rejected,                  
                        "chosen_score": chosen_score,
                        "rejected_score": rej_score,
                    })

            history.append({
                "user": turn["user_utterance"][0],
                "assistant": chosen_txt
            })

print(f"✅ Total examples to embed: {len(output_records)} (aggregate={aggregate_rejected})")

@torch.no_grad()
def extract_embeddings_single_sample(model, head, tokenizer, user_data, save_prefix="emb_output"):
    all_data = user_data

    user_to_samples: Dict[str, List[Dict]] = {}
    for sample in all_data:  
        user_id = sample["user_id"]
        user_to_samples.setdefault(user_id, []).append(sample)

    all_embeddings = []
    metadata_per_user = {}
    current_idx = 0

    for user_id, samples in tqdm(user_to_samples.items(), desc="Processing users"):
        metadata_per_user[user_id] = []
        for sample in samples:

            conv_chosen = sample["conv"] + [{"role": "assistant", "content": sample["chosen"]}]
            conv_rejected = sample["conv"] + [{"role": "assistant", "content": sample["rejected"]}]

            pos_text = tokenizer.apply_chat_template(conv_chosen, tokenize=True, return_tensors="pt").to(model.device)
            neg_text = tokenizer.apply_chat_template(conv_rejected, tokenize=True, return_tensors="pt").to(model.device)

            outputs_pos = model(pos_text, output_hidden_states=True)
            hidden_pos = outputs_pos.hidden_states[-1]
            pos_score = outputs_pos.logits[0][0].item()
            last_emb_pos = hidden_pos[0, -1].to("cpu")  
            cls_score = head(last_emb_pos.unsqueeze(0).to(model.device)).item()
            if abs(cls_score - pos_score) > 1e-3:
                print("positive score mismatch:", cls_score, pos_score)
            all_embeddings.append(last_emb_pos)

            chosen_idx = current_idx
            current_idx += 1

            outputs_neg = model(neg_text, output_hidden_states=True)
            hidden_neg = outputs_neg.hidden_states[-1]
            neg_score = outputs_neg.logits[0][0].item()
            last_emb_neg = hidden_neg[0, -1].to("cpu")  
            cls_score = head(last_emb_neg.unsqueeze(0).to(model.device)).item()
            if abs(cls_score - neg_score) > 1e-3:
                print("negative score mismatch:", cls_score, neg_score)
            all_embeddings.append(last_emb_neg)

            rejected_idx = current_idx
            current_idx += 1

            metadata_per_user[user_id].append({
                "conversation_id": sample["conversation_id"],
                "prompt": sample["conv"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "chosen_idx": chosen_idx,
                "rejected_idx": rejected_idx,
                "chosen_score": pos_score,
                "rejected_score": neg_score,
            })

    print(all_embeddings[0].shape)
    all_embeddings_tensor = torch.stack(all_embeddings, dim=0)
    print(f"Total embeddings shape: {all_embeddings_tensor.shape}")
    torch.save(all_embeddings_tensor, f"{save_prefix}.pt")

    with open(f"{save_prefix}.json", "w") as f:
        json.dump(metadata_per_user, f, indent=2)

    print(f"✅ Saved {len(all_embeddings)} embeddings to {save_prefix}.pt")
    print(f"✅ Metadata saved to {save_prefix}.json")


from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        user_data=output_records,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()