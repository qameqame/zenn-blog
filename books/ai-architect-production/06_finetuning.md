---
title: "Fine-tuning — LoRAでモデルを自社データに特化させる"
---

## はじめに

[第5章のMLOps](./05_mlops)でCI/CDパイプラインを構築しました。本章では別のアプローチとして、モデル自体を自社データで学習し直す「Fine-tuning」を扱います。

```
【RAG】
質問 → DBを検索 → 検索結果をLLMに渡す → 回答
→ ドキュメントが必要・検索コストがかかる

【Fine-tuning】
質問 → Fine-tuned LLM → 回答
→ モデル自体が知識を持っている・検索不要
```

### RAGとFine-tuningの使い分け

| | RAG | Fine-tuning |
|-|-----|-------------|
| 向いている用途 | 最新情報・社内ドキュメント検索 | 特定のスタイル・形式・専門用語 |
| 知識の更新 | ドキュメントを追加するだけ | 再学習が必要 |
| コスト | API費用 + DB費用 | 学習コスト（一回）|
| ハルシネーション | 少ない（根拠あり） | やや多い |
| 実務での例 | 社内FAQ・規約検索 | カスタマーサポート文体・コード生成 |

> **Gemini APIのFine-tuningについて:**
> 2025年5月以降、Gemini API（無料枠）ではFine-tuningがサポートされなくなりました。
> Vertex AI（有料）では引き続き利用可能です。
> 本チュートリアルでは**Hugging Face + LoRA（完全無料）**を使います。

---

## Fine-tuningの種類

### Full Fine-tuning vs LoRA

```
Full Fine-tuning:
  モデルの全パラメータ（数十億）を更新する
  → 高精度だがGPUが必要・コストが高い
  → 7Bモデルで最低16GB VRAM

LoRA（Low-Rank Adaptation）:
  全パラメータの0.1〜1%だけを追加学習する
  → CPUでも動く・低コスト
  → 精度はFull Fine-tuningに近い
```

LoRAの仕組み：

```
元のモデルの重み行列 W（変更しない）
    ↓
低ランク行列 A × B を追加（これだけ学習する）
    ↓
推論時: W + α × (A × B) で計算
```

---

## 前提条件

- `.venv` が有効化済み
- 最低8GBのRAM（CPUで動かす場合）

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル
└── finetuning/
    ├── prepare_dataset.py    # ★ データセット準備
    ├── train_lora.py         # ★ LoRAファインチューニング
    ├── evaluate.py           # ★ ベースモデルとの比較評価
    └── inference.py          # ★ 学習済みモデルで推論
```

---

## 1. ライブラリのインストール

```bash
pip install transformers datasets peft accelerate torch
pip freeze > requirements.txt
```

> **注意:** `torch` は大きなライブラリ（数GB）です。インストールに時間がかかります。
> CPUのみで動かす場合は `pip install torch --index-url https://download.pytorch.org/whl/cpu` が軽量です。

---

## 2. データセットの準備 — `finetuning/prepare_dataset.py`

今まで作ったpgvectorのドキュメントをFine-tuningデータとして使います。

```python
# finetuning/prepare_dataset.py
"""
Fine-tuning用データセットの準備

pgvectorに格納したドキュメントをもとに、
質問→回答のペアを作成します。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# ── 手動で作成した質問・回答ペア ──────────────────────────────
# Fine-tuningには高品質なデータが最重要
# 実務では100〜1000件程度を手動で作成する
TRAINING_DATA = [
    {
        "instruction": "F1スコアとは何ですか？簡潔に説明してください。",
        "input": "",
        "output": "F1スコアはPrecision（適合率）とRecall（再現率）の調和平均です。"
                  "F1 = 2 × Precision × Recall ÷ (Precision + Recall) で計算します。"
                  "クラス不均衡がある場合に特に有効な評価指標です。",
    },
    {
        "instruction": "scikit-learnでクロスバリデーションを実行するコードを教えてください。",
        "input": "",
        "output": "from sklearn.model_selection import cross_val_score\n"
                  "from sklearn.ensemble import RandomForestClassifier\n\n"
                  "model = RandomForestClassifier()\n"
                  "scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n"
                  "print(f'F1スコア: {scores.mean():.3f} ± {scores.std():.3f}')",
    },
    {
        "instruction": "Pandasで欠損値を処理する主な方法を教えてください。",
        "input": "",
        "output": "主な方法は3つあります。\n"
                  "1. 削除: df.dropna() で欠損値を含む行を削除\n"
                  "2. 補完: df.fillna(値) で特定の値に置換（平均値・中央値など）\n"
                  "3. 補間: df.interpolate() で前後の値から補間\n"
                  "数値列は平均値補完、カテゴリ列は最頻値補完が一般的です。",
    },
    {
        "instruction": "AWSのEC2コストを削減する方法を3つ挙げてください。",
        "input": "",
        "output": "1. スポットインスタンスの活用: オンデマンドより最大90%安く利用できます。"
                  "中断可能なバッチ処理に最適です。\n"
                  "2. リザーブドインスタンス: 1〜3年の契約で最大75%割引になります。\n"
                  "3. 不要リソースの削除: 未使用のEBSボリューム・EIP・スナップショットを定期的に削除します。",
    },
    {
        "instruction": "KubernetesのPodとは何ですか？",
        "input": "",
        "output": "PodはKubernetesの最小デプロイ単位です。\n"
                  "1つ以上のコンテナをまとめたグループで、同じノード上で動作し、\n"
                  "ストレージ・ネットワーク・実行仕様を共有します。\n"
                  "YAMLのマニフェストファイルで定義します。",
    },
    # データを増やす（実務では100件以上推奨）
    {
        "instruction": "機械学習のPrecisionとRecallの違いを教えてください。",
        "input": "",
        "output": "Precision（適合率）: 陽性と予測したもののうち実際に陽性だった割合。"
                  "TP / (TP + FP) で計算します。誤検知を減らしたい場合に重視します。\n"
                  "Recall（再現率）: 実際の陽性のうち正しく検出できた割合。"
                  "TP / (TP + FN) で計算します。見逃しを減らしたい場合に重視します。",
    },
    {
        "instruction": "PandasのDataFrameで特定の列を選択する方法は？",
        "input": "",
        "output": "列の選択方法は複数あります。\n"
                  "単一列: df['列名'] または df.列名\n"
                  "複数列: df[['列名1', '列名2']]\n"
                  "条件付き: df.loc[条件, '列名'] または df.iloc[行番号, 列番号]",
    },
    {
        "instruction": "AWSのS3とEBSの違いを教えてください。",
        "input": "",
        "output": "S3（Simple Storage Service）: オブジェクトストレージ。"
                  "ファイルをキーで管理。スケーラブルで安価。静的ファイル・バックアップ向け。\n"
                  "EBS（Elastic Block Store）: ブロックストレージ。"
                  "EC2インスタンスにアタッチして使うHDD/SSD。"
                  "データベース・OSディスクなどに使用。",
    },
]


def save_dataset(data: list[dict], output_file: str):
    """データセットをJSONL形式で保存する"""
    # 学習用（80%）と検証用（20%）に分割
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # JSONLファイルとして保存
    train_file = output_file.replace(".jsonl", "_train.jsonl")
    val_file = output_file.replace(".jsonl", "_val.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"学習データ: {len(train_data)}件 → {train_file}")
    print(f"検証データ: {len(val_data)}件 → {val_file}")
    return train_file, val_file


def format_prompt(item: dict) -> str:
    """Alpacaフォーマットに変換する"""
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

    print(f"データセット件数: {len(TRAINING_DATA)}")
    print("\n--- サンプル ---")
    print(format_prompt(TRAINING_DATA[0]))

    train_file, val_file = save_dataset(
        TRAINING_DATA,
        "finetuning/dataset.jsonl"
    )
    print("\nデータセット準備完了")
```

```bash
mkdir finetuning
python finetuning/prepare_dataset.py
```

---

## 3. LoRAファインチューニング — `finetuning/train_lora.py`

Vertex AIを使う場合はGemini 2.5 FlashのFine-tuningが可能ですが、本チュートリアルでは無料で動くHugging Face + LoRAを使います。

```python
# finetuning/train_lora.py
"""
LoRAを使ったFine-tuning

使用モデル: microsoft/phi-2（2.7B、CPUでも動く軽量モデル）
LoRAの設定: rank=8、alpha=32
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


# ── 設定 ─────────────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"    # 2.7Bの軽量モデル（CPUで動く）
OUTPUT_DIR = "finetuning/lora_output"
MAX_LENGTH = 512


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    """JSONLファイルからデータセットを読み込む"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_prompt(item: dict) -> str:
    """Alpacaフォーマットに変換する"""
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
    print("=== LoRA Fine-tuning 開始 ===\n")

    # ── 1. モデルとトークナイザーの読み込み ──────────────────
    print(f"モデル読み込み中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,        # CPUの場合はfloat32（torch_dtypeは廃止）
        trust_remote_code=True,
    )

    # ── 2. LoRAの設定 ─────────────────────────────────────────
    # rank（r）: 低ランク行列の次元数。大きいほど表現力が上がるがメモリが増える
    # alpha: スケーリング係数。通常はrankの2〜4倍
    # target_modules: LoRAを適用する層（Attention層が一般的）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # ランク（低いほど軽量）
        lora_alpha=32,          # スケーリング係数
        lora_dropout=0.1,       # ドロップアウト
        target_modules=["q_proj", "v_proj"],  # Attention層に適用
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # => trainable params: 約1.3M / total: 2.7B (約0.05%)

    # ── 3. データセットの準備 ─────────────────────────────────
    print("\nデータセット読み込み中...")
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

    # ── 4. 学習設定 ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,           # エポック数
        per_device_train_batch_size=1, # CPUの場合は1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, # 実効バッチサイズ = 4
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",              # WandBなど不要
        use_cpu=True,                  # CPUで動かす（no_cudaは廃止）
    )

    # ── 5. 学習実行 ──────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n学習開始...")
    print("（CPUで動かす場合、数十分〜数時間かかります）")
    trainer.train()

    # ── 6. モデルの保存 ──────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nモデルを保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

```bash
python finetuning/train_lora.py
```

> **CPUで動かす場合の注意:**
> 学習には数十分〜数時間かかります。
> 少ないデータ（8件）では効果が限定的です。実務では100件以上を用意してください。

---

## 4. 推論 — `finetuning/inference.py`

Fine-tuningしたモデルで推論します。

```python
# finetuning/inference.py
"""
Fine-tuningしたモデルで推論する

ベースモデルとFine-tuningモデルの回答を比較します。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "microsoft/phi-2"
LORA_DIR = "finetuning/lora_output"


def load_base_model():
    """ベースモデルを読み込む"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,        # CPUの場合はfloat32（torch_dtypeは廃止）
        trust_remote_code=True,
    )
    return tokenizer, model


def load_finetuned_model():
    """Fine-tuningしたモデルを読み込む"""
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,        # CPUの場合はfloat32（torch_dtypeは廃止）
        trust_remote_code=True,
    )
    # LoRAの重みを読み込む
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    return tokenizer, model


def generate(tokenizer, model, instruction: str, max_new_tokens: int = 200) -> str:
    """テキストを生成する"""
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

    # 入力部分を除いた生成テキストだけを返す
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    test_questions = [
        "F1スコアとは何ですか？",
        "Pandasで欠損値を処理する方法は？",
    ]

    print("=== ベースモデル vs Fine-tuningモデル 比較 ===\n")

    print("ベースモデルを読み込み中...")
    base_tokenizer, base_model = load_base_model()

    print("Fine-tuningモデルを読み込み中...")
    ft_tokenizer, ft_model = load_finetuned_model()

    for question in test_questions:
        print(f"\n質問: {question}")
        print("-" * 50)

        base_answer = generate(base_tokenizer, base_model, question)
        print(f"【ベースモデル】\n{base_answer}")

        ft_answer = generate(ft_tokenizer, ft_model, question)
        print(f"\n【Fine-tuningモデル】\n{ft_answer}")
        print()
```

```bash
python finetuning/inference.py
```

---

## 5. Fine-tuningのデータ品質ガイドライン

Fine-tuningで最も重要なのはデータの品質です。

### 良いデータの条件

```
✅ 一貫性: 同じ質問には同じスタイルで答える
✅ 正確性: 間違った情報を含まない
✅ 多様性: 様々なパターンの質問を含む
✅ 簡潔性: 不要な情報を含まない
✅ 量: 最低100件、理想は1000件以上
```

### データ量とスコアの目安

| データ件数 | 期待できる改善 |
|-----------|--------------|
| 10〜50件 | わずかな変化 |
| 100〜500件 | スタイル・形式の改善が明確に |
| 1000件以上 | 専門知識・用語の定着 |
| 10000件以上 | 特定ドメインの専門家レベル |

---

## 6. Gemini Fine-tuning（有料・Vertex AI）

無料枠ではできませんが、参考として記載します。

```python
# Vertex AIを使う場合（有料）
# 事前に: gcloud auth login / gcloud config set project YOUR_PROJECT

import vertexai
from vertexai.tuning import sft

vertexai.init(project="YOUR_PROJECT", location="us-central1")

# Fine-tuningジョブの作成
sft_tuning_job = sft.train(
    source_model="gemini-2.5-flash",
    train_dataset="gs://your-bucket/train.jsonl",
    validation_dataset="gs://your-bucket/val.jsonl",
    epochs=3,
    learning_rate_multiplier=1.0,
    tuned_model_display_name="my-finetuned-model",
)

print(f"ジョブID: {sft_tuning_job.name}")
print("完了まで数時間かかります")
```

データ形式（JSONL）:
```json
{"contents": [{"role": "user", "parts": [{"text": "F1スコアとは？"}]}, {"role": "model", "parts": [{"text": "F1スコアはPrecisionとRecallの調和平均です。"}]}]}
```

---

## 6. 実行結果の読み方

今回の実行結果から確認できること：

| 指標 | 値 | 意味 |
|------|-----|------|
| `trainable%` | 0.09% | 27億個のうち260万個だけ学習（LoRAの効果） |
| `eval_loss` | 1.802 → 1.785 | エポックごとに改善 → 正しく学習できている |
| `train_runtime` | 109秒 | CPUで2分以内に完了 |

**推論結果について：** phi-2は英語中心のモデルのため日本語の回答品質に限界があります。Fine-tuningモデルでPandasの `fillna()` が出てきたように、わずか8件でもデータの傾向は学習できています。日本語品質を上げるには以下のいずれかが有効です。

```python
# 日本語対応モデルに変更する場合
MODEL_NAME = "llm-jp/llm-jp-3-1.8b"   # 日本語特化・1.8B
# または
MODEL_NAME = "cyberagent/open-calm-3b"  # 日本語・3B
```

データ量を100件以上に増やすことも同様に有効です。

---

## 8. .gitignoreの設定

モデルの重みファイルはGitで管理すべきではありません。リポジトリが数GB肥大化し、バイナリファイルは差分管理が意味を持たないためです。

```bash
cat >> .gitignore << 'EOF'
# Fine-tuning outputs
finetuning/lora_output/
finetuning/*.jsonl
EOF
```

すでにステージングされている場合はキャッシュから削除：

```bash
git rm -r --cached finetuning/lora_output/
git add .gitignore
git commit -m "chore: ignore fine-tuning model outputs"
git push origin main
```

> **モデルの保存先:** 本番運用では [Hugging Face Hub](https://huggingface.co) にアップロードして管理します。`model.push_to_hub("your-username/my-lora-model")` で公開できます。

---

## 9. よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `unexpected keyword argument 'no_cuda'` | 旧パラメータ名 | `use_cpu=True` に変更 |
| `torch_dtype is deprecated` | 旧パラメータ名 | `dtype=torch.float32` に変更 |
| `CUDA out of memory` | GPUメモリ不足 | `per_device_train_batch_size=1` に減らす |
| `ModuleNotFoundError: peft` | インストール忘れ | `pip install peft` |
| 学習が遅すぎる | CPUで動かしている | GPUを使う or データを減らす |
| 回答品質が変わらない | データが少なすぎる or 英語モデル | 100件以上のデータ or 日本語モデルに変更 |
| `ValueError: target_modules` | モデルの層名が違う | `model.named_modules()` で層名を確認 |

---

## 次のステップ

- **[第7章 まとめ](./07_summary)** — AIアーキテクトとして次のステップへ
- **RAG + Fine-tuning の組み合わせ** — Fine-tuningで形式を学習し、RAGで知識を補完するハイブリッドアーキテクチャ
- **マルチエージェント** — 複数のAgentが協調するシステム設計

---

## 参考

- [Hugging Face PEFT ドキュメント](https://huggingface.co/docs/peft)
- [LoRA論文（原著）](https://arxiv.org/abs/2106.09685)
- [Vertex AI Fine-tuning（Gemini）](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning)
- [Alpacaフォーマット](https://github.com/tatsu-lab/stanford_alpaca)
