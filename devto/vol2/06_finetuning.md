---
title: Fine-tuning — Domain-Specializing Models with LoRA
published: false
tags: ai, mlops, llm, python
series: Production Operations Guide for AI Architects
---

## Introduction

In [Chapter 5 (MLOps)](https://dev.to/hiroki-kameyama/mlops-llmops-cicd-pipelines-for-continuous-quality-assurance-46d1), we built a CI/CD pipeline. This chapter explores a different approach: fine-tuning — training the model itself on your own data.

```
[RAG]
Question → search DB → pass results to LLM → answer
→ Requires documents, search costs apply

[Fine-tuning]
Question → Fine-tuned LLM → answer
→ Model carries the knowledge itself — no retrieval needed
```

### When to Use RAG vs Fine-tuning

| | RAG | Fine-tuning |
|-|-----|-------------|
| Best for | Latest info, internal document search | Specific styles, formats, specialized vocabulary |
| Knowledge updates | Just add documents | Requires retraining |
| Cost | API cost + DB cost | Training cost (one-time) |
| Hallucination | Less (grounded) | Somewhat more |
| Practical examples | Internal FAQ, policy search | Customer support tone, code generation |

> **On Gemini API Fine-tuning:**
> Since May 2025, fine-tuning is no longer available on the Gemini API free tier.
> Vertex AI (paid) still supports it.
> This tutorial uses **Hugging Face + LoRA (completely free)**.

---

## Types of Fine-tuning

### Full Fine-tuning vs LoRA

```
Full Fine-tuning:
  Updates all parameters (billions) of the model
  → High accuracy but requires GPU — expensive
  → 7B model needs minimum 16GB VRAM

LoRA (Low-Rank Adaptation):
  Trains only 0.1–1% of all parameters
  → Works on CPU, low cost
  → Accuracy close to full fine-tuning
```

How LoRA works:

```
Original model weight matrix W (unchanged)
    ↓
Add low-rank matrices A × B (only these are trained)
    ↓
At inference: compute W + α × (A × B)
```

---

## Directory Structure

```
pgvector-tutorial/
├── existing files
└── finetuning/
    ├── prepare_dataset.py    # ★ Dataset preparation
    ├── train_lora.py         # ★ LoRA fine-tuning
    ├── evaluate.py           # ★ Comparison with base model
    └── inference.py          # ★ Inference with trained model
```

---

## 1. Install Libraries

```bash
pip install transformers datasets peft accelerate torch
pip freeze > requirements.txt
```

> **Note:** `torch` is a large library (several GB). Installation takes time.
> For CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 2. Dataset Preparation — `finetuning/prepare_dataset.py`

We'll use the pgvector documents created earlier as fine-tuning data.

```python
# finetuning/prepare_dataset.py
"""
Fine-tuning dataset preparation

Create question → answer pairs from pgvector documents.
"""
import json
import os

# ── Manually created Q&A pairs ────────────────────────────────
# Data quality is paramount for fine-tuning.
# In production, create 100–1000 high-quality pairs by hand.
TRAINING_DATA = [
    {
        "instruction": "What is the F1 score? Explain briefly.",
        "input": "",
        "output": "The F1 score is the harmonic mean of Precision (positive predictive value) "
                  "and Recall (sensitivity). "
                  "F1 = 2 × Precision × Recall ÷ (Precision + Recall). "
                  "It's especially useful when class imbalance is present.",
    },
    {
        "instruction": "Show me code for cross-validation with scikit-learn.",
        "input": "",
        "output": "from sklearn.model_selection import cross_val_score\n"
                  "from sklearn.ensemble import RandomForestClassifier\n\n"
                  "model = RandomForestClassifier()\n"
                  "scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n"
                  "print(f'F1 Score: {scores.mean():.3f} ± {scores.std():.3f}')",
    },
    {
        "instruction": "What are the main methods for handling missing values in Pandas?",
        "input": "",
        "output": "There are three main approaches:\n"
                  "1. Drop: df.dropna() removes rows with missing values\n"
                  "2. Fill: df.fillna(value) replaces with a specified value (mean, median, etc.)\n"
                  "3. Interpolate: df.interpolate() fills using surrounding values\n"
                  "Numeric columns typically use mean imputation; categorical columns use mode.",
    },
    {
        "instruction": "Give me three ways to reduce EC2 costs on AWS.",
        "input": "",
        "output": "1. Use Spot Instances: up to 90% cheaper than On-Demand. "
                  "Best for interruptible batch workloads.\n"
                  "2. Reserved Instances: 1–3 year commitments for up to 75% discount.\n"
                  "3. Delete unused resources: regularly remove idle EBS volumes, EIPs, and snapshots.",
    },
    {
        "instruction": "What is a Kubernetes Pod?",
        "input": "",
        "output": "A Pod is the smallest deployable unit in Kubernetes.\n"
                  "It groups one or more containers that run on the same node, "
                  "sharing storage, networking, and execution specifications.\n"
                  "Pods are defined using YAML manifest files.",
    },
    {
        "instruction": "What is the difference between Precision and Recall?",
        "input": "",
        "output": "Precision (positive predictive value): of all predicted positives, "
                  "the fraction that are actually positive. TP / (TP + FP). "
                  "Prioritize when false positives are costly.\n"
                  "Recall (sensitivity): of all actual positives, "
                  "the fraction correctly detected. TP / (TP + FN). "
                  "Prioritize when false negatives are costly.",
    },
    {
        "instruction": "How do you select specific columns from a Pandas DataFrame?",
        "input": "",
        "output": "Several approaches are available:\n"
                  "Single column: df['column_name'] or df.column_name\n"
                  "Multiple columns: df[['col1', 'col2']]\n"
                  "Label-based: df.loc[condition, 'column']\n"
                  "Integer-based: df.iloc[row_index, col_index]",
    },
    {
        "instruction": "What is the difference between AWS S3 and EBS?",
        "input": "",
        "output": "S3 (Simple Storage Service): Object storage. "
                  "Manages files by key. Scalable and inexpensive. Ideal for static files and backups.\n"
                  "EBS (Elastic Block Store): Block storage. "
                  "Attached to EC2 instances like an HDD/SSD. "
                  "Used for databases and OS disks.",
    },
]


def save_dataset(data: list[dict], output_file: str):
    """Save dataset in JSONL format with 80/20 train/val split."""
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_file = output_file.replace(".jsonl", "_train.jsonl")
    val_file = output_file.replace(".jsonl", "_val.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Training data: {len(train_data)} samples → {train_file}")
    print(f"Validation data: {len(val_data)} samples → {val_file}")
    return train_file, val_file


def format_prompt(item: dict) -> str:
    """Convert to Alpaca format."""
    if item["input"]:
        return (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Input:\n{item['input']}\n\n"
            f"### Response:\n{item['output']}"
        )
    return (
        f"### Instruction:\n{item['instruction']}\n\n"
        f"### Response:\n{item['output']}"
    )


if __name__ == "__main__":
    os.makedirs("finetuning", exist_ok=True)

    print(f"Dataset size: {len(TRAINING_DATA)} samples")
    print("\n--- Sample ---")
    print(format_prompt(TRAINING_DATA[0]))

    train_file, val_file = save_dataset(
        TRAINING_DATA,
        "finetuning/dataset.jsonl"
    )
    print("\nDataset preparation complete")
```

```bash
mkdir finetuning
python finetuning/prepare_dataset.py
```

---

## 3. LoRA Fine-tuning — `finetuning/train_lora.py`

```python
# finetuning/train_lora.py
"""
LoRA Fine-tuning

Model: microsoft/phi-2 (2.7B, runs on CPU)
LoRA config: rank=8, alpha=32
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "finetuning/lora_output"
MAX_LENGTH = 512


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_prompt(item: dict) -> str:
    if item.get("input"):
        return (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Input:\n{item['input']}\n\n"
            f"### Response:\n{item['output']}"
        )
    return (
        f"### Instruction:\n{item['instruction']}\n\n"
        f"### Response:\n{item['output']}"
    )


def main():
    print("=== LoRA Fine-tuning Start ===\n")

    # ── 1. Load model and tokenizer ───────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,      # float32 for CPU (torch_dtype is deprecated)
        trust_remote_code=True,
    )

    # ── 2. LoRA configuration ─────────────────────────────────
    # r (rank): dimension of low-rank matrices. Larger = more expressive but more memory
    # alpha: scaling factor, typically 2–4× rank
    # target_modules: layers to apply LoRA (usually Attention layers)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # => trainable params: ~1.3M / total: 2.7B (~0.05%)

    # ── 3. Prepare datasets ───────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = load_dataset_from_jsonl("finetuning/dataset_train.jsonl")
    val_dataset = load_dataset_from_jsonl("finetuning/dataset_val.jsonl")

    def tokenize_function(examples):
        texts = [format_prompt({
            "instruction": inst,
            "input": inp,
            "output": out,
        }) for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        )]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    # ── 4. Training configuration ─────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,    # 1 for CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,    # effective batch size = 4
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        use_cpu=True,                     # CPU mode (no_cuda is deprecated)
    )

    # ── 5. Run training ───────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\nTraining started...")
    print("(On CPU, this may take tens of minutes to hours)")
    trainer.train()

    # ── 6. Save model ─────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

```bash
python finetuning/train_lora.py
```

---

## 4. Inference — `finetuning/inference.py`

Run inference with the fine-tuned model and compare it to the base model.

```python
# finetuning/inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "microsoft/phi-2"
LORA_DIR = "finetuning/lora_output"


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model


def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    return tokenizer, model


def generate(tokenizer, model, instruction: str, max_new_tokens: int = 200) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    test_questions = [
        "What is the F1 score?",
        "How do you handle missing values in Pandas?",
    ]

    print("=== Base Model vs Fine-tuned Model Comparison ===\n")

    print("Loading base model...")
    base_tokenizer, base_model = load_base_model()

    print("Loading fine-tuned model...")
    ft_tokenizer, ft_model = load_finetuned_model()

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        base_answer = generate(base_tokenizer, base_model, question)
        print(f"[Base Model]\n{base_answer}")

        ft_answer = generate(ft_tokenizer, ft_model, question)
        print(f"\n[Fine-tuned Model]\n{ft_answer}")
```

```bash
python finetuning/inference.py
```

---

## 5. Data Quality Guidelines

Data quality is the most important factor in fine-tuning.

### Characteristics of Good Data

```
✅ Consistency: Same question → same style of answer
✅ Accuracy: No incorrect information
✅ Diversity: Cover a wide variety of question patterns
✅ Conciseness: No unnecessary information
✅ Quantity: Minimum 100 samples, ideally 1000+
```

### Data Size vs Expected Improvement

| Dataset Size | Expected Improvement |
|-------------|---------------------|
| 10–50 samples | Minimal change |
| 100–500 samples | Clear improvement in style and format |
| 1,000+ samples | Domain knowledge and terminology retention |
| 10,000+ samples | Expert-level domain specialization |

---

## 6. Reading the Training Results

What the output tells you:

| Metric | Value | Meaning |
|--------|-------|---------|
| `trainable%` | 0.09% | Only 2.6M of 2.7B parameters trained (LoRA effect) |
| `eval_loss` | 1.802 → 1.785 | Improving each epoch → learning correctly |
| `train_runtime` | 109s | Completed in under 2 minutes on CPU |

---

## 7. Add to `.gitignore`

Model weight files should not be version-controlled — they can bloat the repository by several GB.

```bash
cat >> .gitignore << 'EOF'
# Fine-tuning outputs
finetuning/lora_output/
finetuning/*.jsonl
EOF
```

For production use, upload to [Hugging Face Hub](https://huggingface.co):
```python
model.push_to_hub("your-username/my-lora-model")
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `unexpected keyword argument 'no_cuda'` | Old parameter name | Use `use_cpu=True` |
| `torch_dtype is deprecated` | Old parameter name | Use `dtype=torch.float32` |
| `CUDA out of memory` | Insufficient GPU memory | Reduce `per_device_train_batch_size` to 1 |
| `ModuleNotFoundError: peft` | Not installed | `pip install peft` |
| Training too slow | Running on CPU | Use GPU or reduce dataset |
| Answer quality unchanged | Too little data | Use 100+ samples or switch to a Japanese model |

---

## Next Steps

- **[Chapter 7: Multi-Agent]** — Design systems where Orchestrators and Workers collaborate
- **RAG + Fine-tuning hybrid** — Fine-tune for style, use RAG for knowledge
- **Vertex AI Fine-tuning (paid)** — For Gemini 2.5 Flash fine-tuning on your own data
