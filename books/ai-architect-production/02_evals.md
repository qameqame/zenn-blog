---
title: "Evals — RAGの回答品質を自動測定する"
---

## はじめに

[前作のRAG実装](https://zenn.dev/hkame/books/ai-architect-rag/viewer/02_rag)では動くシステムを作りました。しかし「本当に正しく答えられているか」は手動で確認するしかありませんでした。

```
【今まで】手動確認
「F1スコアはどう計算しますか？」と聞いて目で見て確認

【今回 Evals】
テストケースを用意して自動で品質スコアを測定
```

Evalsとは「評価データセット（質問と期待する回答）」を用意して、システムの回答を自動採点する仕組みです。

---

## 評価の3つの観点

RAGシステムの評価は大きく3つの観点があります。

| 観点 | 意味 | 測定すること |
|------|------|------------|
| **Faithfulness** | 忠実性 | 検索結果に基づいて回答しているか（ハルシネーションがないか） |
| **Answer Relevancy** | 回答の関連性 | 質問に対して適切な回答をしているか |
| **Context Recall** | 検索の再現率 | 正解を含むドキュメントを正しく検索できているか |

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル（01〜13）
│
├── evals/
│   ├── dataset.py        # ★ 評価データセットの定義
│   ├── eval_rag.py       # ★ RAGの評価
│   ├── eval_agent.py     # ★ Agentの評価
│   └── report.py         # ★ 評価レポートの生成
```

---

## 1. ライブラリのインストール

```bash
pip install pandas tabulate
pip freeze > requirements.txt
```

---

## 2. 評価データセットの定義 — `evals/dataset.py`

評価データセットは「質問・期待する回答・期待する参照ドキュメント」のセットです。

```python
# evals/dataset.py

# 評価データセット
# 各エントリは「質問・期待する回答の要素・期待する検索ドキュメント」で構成
EVAL_DATASET = [
    {
        "id": "eval_001",
        "question": "F1スコアはどう計算しますか？",
        "expected_answer_keywords": ["Precision", "Recall", "調和平均", "2"],
        "expected_docs": ["機械学習モデルの評価指標"],
        "category": "ML",
    },
    {
        "id": "eval_002",
        "question": "scikit-learnでモデルを評価する方法は？",
        "expected_answer_keywords": ["cross_val_score", "classification_report", "scikit-learn"],
        "expected_docs": ["scikit-learnによるモデル評価"],
        "category": "ML",
    },
    {
        "id": "eval_003",
        "question": "AWSのコストを下げる方法は？",
        "expected_answer_keywords": ["EC2", "スポットインスタンス", "コスト"],
        "expected_docs": ["AWSコスト最適化の実践"],
        "category": "Cloud",
    },
    {
        "id": "eval_004",
        "question": "Pandasで欠損値を処理するには？",
        "expected_answer_keywords": ["欠損値", "DataFrame", "Pandas"],
        "expected_docs": ["Pandasによるデータ前処理"],
        "category": "Python",
    },
    {
        "id": "eval_005",
        "question": "Kubernetesのマニフェストファイルの書き方は？",
        "expected_answer_keywords": ["YAML", "Pod", "Kubernetes"],
        "expected_docs": ["Kubernetes Podの基本"],
        "category": "Cloud",
    },
]
```

---

## 3. RAGの評価 — `evals/eval_rag.py`

```python
# evals/eval_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
from evals.dataset import EVAL_DATASET

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


def get_query_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


def search(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    cur.execute("""
        SELECT title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "similarity": round(r[2], 4)}
        for r in rows
    ]


def rag_answer(question: str) -> tuple[str, list[dict]]:
    """RAGで回答を生成し、使用したドキュメントも返す"""
    docs = search(question, top_k=3)
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])
    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答（参考ドキュメントに基づいて簡潔に）"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text, docs
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise


# ══════════════════════════════════════════
# 評価関数
# ══════════════════════════════════════════

def eval_context_recall(retrieved_docs: list[dict], expected_docs: list[str]) -> float:
    """
    Context Recall: 期待するドキュメントが検索結果に含まれているか

    スコア = 期待するドキュメントのうち実際に取得できた割合
    """
    retrieved_titles = [d["title"] for d in retrieved_docs]
    hit = sum(1 for expected in expected_docs if expected in retrieved_titles)
    return hit / len(expected_docs) if expected_docs else 0.0


def eval_answer_relevancy(answer: str, keywords: list[str]) -> float:
    """
    Answer Relevancy: 回答に期待するキーワードが含まれているか

    スコア = 期待するキーワードのうち回答に含まれた割合
    """
    hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hit / len(keywords) if keywords else 0.0


def eval_faithfulness(answer: str, retrieved_docs: list[dict]) -> float:
    """
    Faithfulness: 回答が検索結果に基づいているか
    LLMを使って採点する（LLM-as-a-Judge パターン）

    スコア = 0.0〜1.0（LLMが採点）
    """
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in retrieved_docs])
    prompt = f"""以下のコンテキストと回答を評価してください。

# コンテキスト（検索で取得したドキュメント）
{context}

# 回答
{answer}

評価基準:
- 回答がコンテキストの内容に基づいているか
- コンテキストにない情報を勝手に追加していないか（ハルシネーション）

0.0〜1.0のスコアだけを返してください。説明不要。数値のみ。"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            score_text = response.text.strip()
            return float(score_text)
        except (ValueError, Exception) as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                return 0.5  # 評価失敗時はデフォルト値


def run_eval():
    """全評価データセットに対してRAGを評価する"""
    results = []

    print("RAG評価を開始します...")
    print("=" * 60)

    for item in EVAL_DATASET:
        print(f"\n[{item['id']}] {item['question']}")

        # RAGで回答を生成
        answer, retrieved_docs = rag_answer(item["question"])
        time.sleep(2)  # レート制限対策

        # 各指標を評価
        context_recall   = eval_context_recall(retrieved_docs, item["expected_docs"])
        answer_relevancy = eval_answer_relevancy(answer, item["expected_answer_keywords"])
        faithfulness     = eval_faithfulness(answer, retrieved_docs)
        time.sleep(2)

        # 総合スコア（3指標の平均）
        overall = (context_recall + answer_relevancy + faithfulness) / 3

        result = {
            "id":               item["id"],
            "question":         item["question"][:30] + "...",
            "context_recall":   round(context_recall, 2),
            "answer_relevancy": round(answer_relevancy, 2),
            "faithfulness":     round(faithfulness, 2),
            "overall":          round(overall, 2),
        }
        results.append(result)

        print(f"  Context Recall:   {context_recall:.2f}")
        print(f"  Answer Relevancy: {answer_relevancy:.2f}")
        print(f"  Faithfulness:     {faithfulness:.2f}")
        print(f"  Overall:          {overall:.2f}")

    return results


if __name__ == "__main__":
    results = run_eval()

    print("\n" + "=" * 60)
    print("評価結果サマリー")
    print("=" * 60)

    avg_recall    = sum(r["context_recall"]   for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_faith     = sum(r["faithfulness"]     for r in results) / len(results)
    avg_overall   = sum(r["overall"]          for r in results) / len(results)

    print(f"Context Recall:   {avg_recall:.2f}")
    print(f"Answer Relevancy: {avg_relevancy:.2f}")
    print(f"Faithfulness:     {avg_faith:.2f}")
    print(f"Overall:          {avg_overall:.2f}")
```

```bash
python evals/eval_rag.py
```

実行結果の例：

```
RAG評価を開始します...
============================================================

[eval_001] F1スコアはどう計算しますか？
  Context Recall:   1.00
  Answer Relevancy: 1.00
  Faithfulness:     0.92
  Overall:          0.97

[eval_002] scikit-learnでモデルを評価する方法は？
  Context Recall:   1.00
  Answer Relevancy: 0.75
  Faithfulness:     0.88
  Overall:          0.88

============================================================
評価結果サマリー
============================================================
Context Recall:   0.95
Answer Relevancy: 0.85
Faithfulness:     0.90
Overall:          0.90
```

---

## 4. 評価の読み方

| スコア | 意味 |
|--------|------|
| 0.9以上 | 優秀。本番運用レベル |
| 0.7〜0.9 | 良好。改善の余地あり |
| 0.5〜0.7 | 要改善。ドキュメントや検索設定を見直す |
| 0.5未満 | 問題あり。設計を再検討 |

### 各指標が低いときの対処法

**Context Recall が低い場合**
→ 検索が期待するドキュメントを取得できていない
→ `top_k` を増やす・ドキュメントのチャンク設計を見直す・メタデータフィルタを追加する

**Answer Relevancy が低い場合**
→ 回答が質問に対してズレている
→ プロンプトを改善する・システムプロンプトを追加する

**Faithfulness が低い場合**
→ 検索結果にない情報を回答に含めている（ハルシネーション）
→ プロンプトに「ドキュメントにない情報は答えないこと」を明示する

---

## 5. LLM-as-a-Judgeパターンについて

`eval_faithfulness()` でLLM自身に採点させる手法を **LLM-as-a-Judge** と呼びます。

```
通常の評価:
  人間が正解を定義 → ルールベースで採点
  → 速い・安定 → 複雑な判断が難しい

LLM-as-a-Judge:
  LLMが採点基準を理解して採点
  → 複雑な判断も可能 → コストがかかる・採点がブレることがある
```

今回は2つを組み合わせています。

| 指標 | 手法 | 理由 |
|------|------|------|
| Context Recall | ルールベース（タイトル一致） | 明確な正解がある |
| Answer Relevancy | ルールベース（キーワード一致） | 明確な正解がある |
| Faithfulness | LLM-as-a-Judge | ハルシネーションの判断は複雑 |

---

## 6. よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `ValueError: could not convert string to float` | LLMが数値以外を返した | プロンプトを強化・デフォルト値で対応 |
| `429 RESOURCE_EXHAUSTED` | レート制限 | `time.sleep()` で待機時間を増やす |
| スコアが常に0 | キーワードの表記ゆれ | expected_answer_keywordsを見直す |

---

## 次のステップ

- **[第3章 Observability](./03_observability)** — LangfuseでRAGの各ステップをトレースして可視化する
- **RAGASの導入** — `pip install ragas` でより高度な評価フレームワークを使う
- **継続的評価（CI/CD）** — 第5章のMLOpsでGitHub Actionsと組み合わせる

---

## 参考

- [RAGAS — RAG評価フレームワーク](https://docs.ragas.io)
- [LangSmith — トレーシング・評価ツール](https://smith.langchain.com)
- [Langfuse — OSS版Observabilityツール](https://langfuse.com)
