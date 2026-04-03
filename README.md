
<a name="readme-top"></a>

<div align="center">
  <h1 align="center">One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment</h1>
</div>

<div align="center">

  <!-- Paper Link -->
  <a href="https://www.arxiv.org/pdf/2601.18731">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper" height="22">
  </a>

  <a href="https://huggingface.co/papers/2601.18731">
    <img src="https://img.shields.io/badge/HuggingFace-Papers-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Papers"  height="22">
  </a>

  <!-- HuggingFace Models -->
  <a href="https://huggingface.co/collections/ModalityDance/mrm">
    <img src="https://img.shields.io/badge/HuggingFace-Models-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Models" height="22">
  </a>


</div>


<div align="center">
  <figure>
    <img src="./assets/overview.png" alt="Overview" style="width: 50%; height: auto;">
    <br>
    <figcaption><em>Quick Overview of Meta Reward Modeling.</em></figcaption>
  </figure>
</div>

Welcome to Meta Reward Modeling (MRM)! 👋
MRM is a research-oriented framework for personalized reward modeling and alignment of Large Language Models (LLMs). It is designed to study how reward models can efficiently adapt to diverse user preferences under sparse feedback and generalize to unseen users. The framework follows a clean and modular design, making it easy to prototype, extend, and evaluate personalized alignment methods.

This project provides a full implementation of MRM, where each user’s preference learning is treated as a separate task. It includes MAML-style training pipelines, few-shot user adaptation, robust optimization objectives for hard-to-learn users, and reproducible evaluation scripts for user-level performance analysis.

### 🪐 Key Features

🧭 **Meta-Learned Personalized Reward Initialization**: Treats each user as a separate task and learns a shared reward initialization that can quickly adapt to new users with only a few preference examples.

🌌 **Robust Training for Diverse User Preferences**: Uses a robust personalization objective that focuses more on hard-to-model users, improving performance consistency across diverse and long-tail preferences.

🧩 **Lightweight and Modular Reward Design**: Represents user rewards with low-dimensional adaptive weights over shared components, enabling efficient personalization, clean ablations, and easy extension.


## 🔥 News 

<div style="max-height: 240px; overflow-y: auto;">
  
- **[2026.04]** ✨✨MRM is accepted by **SIGIR 2026** !!!✨✨

- **[2026.01]** 🎉🎉Initial release of the project.

</div>


## 📑 Table of Contents <span id="table-of-contents"></span>


* <a href='#quick-start'>🚀 Quick Start</a>
  * <a href='#installation'>Installation</a>
  * <a href='#data'>Data</a>
  * <a href='#running'>Running</a>
* <a href='#usage-example'>🧪 Usage Example</a>
* <a href='#how-it-works'>✨ How It Works</a>
* <a href='#acknowledgements'>🌱 Acknowledgements</a>
* <a href='#citation'>📚 Citation</a>



## 🚀 Quick Start <span id="quick-start"></span>


### 1. Installation <span id="installation"></span>

The code is tested on Python 3.10.0, PyTorch 2.4.0 and CUDA 12.5. 

You can create a conda environment with the required dependencies using the provided `requirements.txt` file.

```bash 
conda create -n mrm python=3.10 -y
conda activate mrm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Data Preparation <span id="data"></span>

1. The dataset used in the paper is the [PRISM](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) dataset and the [Reddit TLDR](https://huggingface.co/datasets/openai/summarize_from_feedback) dataset.
2. Run the following command to download and preprocess the data to generate the embeddings:

For PRISM:
```bash
python scripts/preprocess_prism.py \
  --model_path Skywork/Skywork-Reward-V2-Llama-3.1-8B \
  --save_prefix data/emb/prism/V2 \
```
For Reddit TLDR:
```bash
python scripts/preprocess_reddit.py \
  --model_path Skywork/Skywork-Reward-V2-Llama-3.1-8B \
  --save_prefix data/emb/reddit/V2
```
> [!NOTE]
> Change the model path to 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2' if you want to use the V1 version of the reward model for preprocessing.

### 3. Running <span id="running"></span>


#### **Training**

After data preparation, you can start training the meta reward model using the following command:
for PRISM:
```bash
bash scripts/train_on_prism.sh
```

for Reddit TLDR:

```bash
bash scripts/train_on_reddit.sh
```
Both scripts will train the model with our default hyperparameters, and evaluate the model on the test set after predefined intervals. Also, logs and model checkpoints will be saved under the `output/` directory.

> [!NOTE]
> 1. Our training pipelines include automatic evaluation and checkpointing, so typically you do not need to run the evaluation script.
> 
> 2. You can modify the hyperparameters in the training scripts as needed. For example, change the 'seen_train_limit' to 100 to replicate the results of Reddit TLDR with 100 training samples per user.
>
> 3. If you want to skip the training phase and directly evaluate with our pretrained checkpoints. Download from [here](https://huggingface.co/collections/ModalityDance/mrm).

#### **Evaluation**

After training, you can evaluate the saved model checkpoints using the following commands:
For PRISM:
```bash
bash scripts/test_on_prism.sh
```
For Reddit TLDR:
```bash
bash scripts/test_on_reddit.sh
```

> [!IMPORTANT]
> 1. As every training run will randomly split the users and samples, please make sure to use the same setting (inner epoch, inner leaerning rate, and random seed) with the training phase when doing evaluation for consistent results.
>
> 2. For our released checkpoints, please refer to the provided inference scripts for the exact hyperparameters used.

## 🧪 Usage Example <span id="usage-example"></span>

This example shows a typical workflow for a **single user**:
1) Encode text pairs with Skywork-Reward-V2-Llama-3.1-8B into embeddings,
2) Adapt the MRM on the user's few-shot examples (update `shared_weight` only),
3) Run inference on new pairs for that same user.

```python
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

```

## ✨ How It Works <span id="how-it-works"></span>

Meta Reward Modeling is a modular framework for personalized reward modeling, designed to learn rewards that can quickly adapt to individual users from limited preference data.
The method separates shared reward structure from user-specific adaptation, enabling few-shot personalization and robust generalization to unseen users.

At a high level, the workflow proceeds as follows:

1. Preference Representation — Each response is scored using shared base reward functions, producing feature-level reward signals that are common across users.

2. Meta-Learned Personalization — A shared initialization of user-specific weights is learned across users. For each user, these weights are adapted with a few gradient steps using their own preference data.

3. Robust Meta Optimization — During training, user-level losses are reweighted to focus more on hard-to-model users, ensuring stable performance across diverse and long-tail preferences.

<div align="center"> <figure> <img src="./assets/method.png" alt="Method Overview" style="width: 50%; height: auto;"> <br> <figcaption><em>Overview of Meta Reward Modeling.</em></figcaption> </figure> </div>


## 🌱 **Acknowledgements** <span id="acknowledgements"></span>

We would like to thank the contributors, open-source projects, and research communities whose work made **Meta Reward Modeling** possible. 

[![Skywork Reward](https://img.shields.io/badge/🤗%20Skywork--Reward--V1-HuggingFace-yellow)](https://huggingface.co/collections/Skywork/skywork-reward-model)
[![Skywork Reward V2](https://img.shields.io/badge/🤗%20Skywork--Reward--V2-HuggingFace-yellow)](https://huggingface.co/collections/Skywork/skywork-reward-v2)
[![PRISM Dataset](https://img.shields.io/badge/🤗%20PRISM-Dataset-green)](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)
[![Reddit TL;DR](https://img.shields.io/badge/🤗%20Reddit--TLDR-Dataset-green)](https://huggingface.co/datasets/openai/summarize_from_feedback)
[![LoRe](https://img.shields.io/badge/GitHub-LoRe-black?logo=github)](https://github.com/facebookresearch/LoRe)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-red)](https://pytorch.org/)
[![🤗 Transformers](https://img.shields.io/badge/🤗%20Transformers-Used-yellow)](https://github.com/huggingface/transformers)

This project is licensed under the **MIT License**. Please refer to the LICENSE file for more details.

### 🔗 Related Projects

<div align="center">

<table>
<tr>
<td align="center">
  <b>🌟 PersonalWAB</b><br/>
  <a href="https://hongrucai.github.io/PersonalWAB/">Project Page</a>
</td>
<td align="center">
  <b>🚀 LoRe</b><br/>
  <a href="https://github.com/facebookresearch/LoRe">GitHub Repo</a>
</td>
<td align="center">
  <b>🔧 SynthesizeMe</b><br/>
  <a href="https://github.com/SALT-NLP/SynthesizeMe">GitHub Repo</a>
</td>
</tr>
</table>

</div>


## 📚 **Citation** <span id="citation"></span>

If you use **Meta Reward Modeling** in your research or applications, please consider citing:

```bibtex
@misc{cai2026adaptsanymetareward,
      title={One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment}, 
      author={Hongru Cai and Yongqi Li and Tiezheng Yu and Fengbin Zhu and Wenjie Wang and Fuli Feng and Wenjie Li},
      year={2026},
      eprint={2601.18731},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.18731}, 
}
```

<div align="center">

<a href="https://github.com/ModalityDance/MRM">
  <img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://github.com/ModalityDance/MRM/issues">
  <img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github" />
</a>

<br/>
⭐ <b>Thank you for visiting Meta Reward Modeling!</b> ⭐

</div>
