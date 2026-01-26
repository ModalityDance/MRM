import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import bt_loss
from train import MRM
from inference import load_ckpt_into_model


@torch.no_grad()
def encode_pairs(model, tokenizer, pairs, device="cuda"):
    model.eval()
    ch, rj = [], []
    for ex in pairs:
        conv = ex["prompt"]
        for key, buf in [("chosen", ch), ("rejected", rj)]:
            ids = tokenizer.apply_chat_template(
                conv + [{"role": "assistant", "content": ex[key]}],
                tokenize=True, return_tensors="pt"
            ).to(device)
            out = model(ids, output_hidden_states=True)
            buf.append(out.hidden_states[-1][0, -1].float().cpu())
    return torch.stack(ch), torch.stack(rj)


def adapt_single_user(base_model, support_ch, support_rj, inner_lr=1e-3, inner_epochs=5, device="cuda"):
    model = deepcopy(base_model).to(device).train()
    opt = torch.optim.Adam([model.shared_weight], lr=inner_lr)
    support_ch, support_rj = support_ch.to(device), support_rj.to(device)
    for _ in range(inner_epochs):
        opt.zero_grad()
        loss = bt_loss(model(support_ch), model(support_rj))
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def infer_on_pairs(model, ch, rj, device="cuda"):
    return model(ch.to(device)), model(rj.to(device))


device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16, device_map=device
)

CKPT_PATH = "ckpt/model.pt"
mrm = MRM(in_dim=4096, hidden_sizes=[2], use_bias=False)
load_ckpt_into_model(mrm, CKPT_PATH, device)

support_pairs = [
    {
        "prompt": [{"role": "user", "content": "TL;DR this post: I tried waking up at 5am for a month and tracked my productivity."}],
        "chosen": "Waking up early helped at first, but long-term productivity depended more on sleep quality than wake-up time.",
        "rejected": "The post is about waking up early and productivity.",
    },
    {
        "prompt": [{"role": "user", "content": "Summarize the main point: I switched from iPhone to Android after 10 years."}],
        "chosen": "The author values customization and battery life more than ecosystem lock-in, which motivated the switch.",
        "rejected": "The author bought a new phone.",
    },
]

sup_ch, sup_rj = encode_pairs(llm, tokenizer, support_pairs, device)
user_mrm = adapt_single_user(mrm, sup_ch, sup_rj, device=device)

test_pairs = [
    {
        "prompt": [{"role": "user", "content": "TL;DR: I quit my job to freelance and here is what I learned in 6 months."}],
        "chosen": "Freelancing offers flexibility but requires strong self-discipline and financial planning to be sustainable.",
        "rejected": "The author talks about quitting a job and freelancing.",
    }
]

test_ch, test_rj = encode_pairs(llm, tokenizer, test_pairs, device)
s_ch, s_rj = infer_on_pairs(user_mrm, test_ch, test_rj, device)

print("reward(chosen)  =", s_ch.tolist())
print("reward(rejected)=", s_rj.tolist())
