---
title: "Observability — Langfuse v4でRAG・Agentをトレースする"
---

## はじめに

[第2章のEvals](./02_evals)で「回答の品質」を測定しました。次は「何が起きているか」を可視化するObservabilityです。

```
【Evals】
回答が正しいかをスコアで測定 → 品質の測定

【Observability】
どのツールを何回呼んだか・何秒かかったか → 動作の可視化
```

今回は **Langfuse v4**（OSSのObservabilityツール）を使います。クラウド版は無料枠あり、セルフホストも可能です。

> **注意: Langfuse v4（2026年3月〜）はAPIが大幅に変わっています。**
> `langfuse_context`・`update_current_observation`・`update_current_trace` は廃止されました。
> 本チュートリアルはv4.9以降に対応しています。

---

## Langfuseでできること

| 機能 | 内容 |
|------|------|
| **トレーシング** | RAG・Agentの各ステップの実行時間・入出力を記録 |
| **コスト管理** | APIの使用量・コストを可視化 |
| **ダッシュボード** | 全体の品質・レイテンシをリアルタイムで確認 |

---

## 前提条件

- 前回のEvalsチュートリアル完了済み
- `.venv` が有効化済み

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル
├── evals/
│   └── ...                # 作成済み
│
└── observability/
    ├── traced_rag.py      # ★ トレース付きRAG（今回追加）
    └── traced_agent.py    # ★ トレース付きAgent（今回追加）
```

---

## Step 1: Langfuseのセットアップ

### 1-1. ライブラリのインストール

```bash
pip install langfuse
pip freeze > requirements.txt
```

### 1-2. Langfuseアカウント作成

1. [cloud.langfuse.com](https://cloud.langfuse.com) にアクセス
2. GitHubアカウントでサインアップ（無料・クレジットカード不要）
3. 新しいプロジェクトを作成
4. 「Settings」→「API Keys」から以下を取得：
   - `LANGFUSE_PUBLIC_KEY`（`pk-lf-...` から始まる）
   - `LANGFUSE_SECRET_KEY`（`sk-lf-...` から始まる）

### 1-3. `.env` に追加

```
# 既存の設定
GEMINI_API_KEY=AIza...
DB_HOST=localhost
...

# Langfuse（新規追加）
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

設定確認：

```bash
grep LANGFUSE .env
```

3行表示されればOKです。

> **⚠️ 重要: `load_dotenv()` より前に `get_client()` を呼ばないこと**
> Langfuseは初期化時に環境変数を読み込みます。`load_dotenv()` の後に `get_client()` を呼ぶ順序を守ってください。

---

## Step 2: トレース付きRAG — `observability/traced_rag.py`

`@observe()` デコレーターを追加するだけで自動的にトレースが記録されます。

```python
# observability/traced_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langfuse import get_client, observe
import time

# ── 必ずload_dotenv()を先に呼ぶ ──────────────────────────────
load_dotenv()
langfuse = get_client()  # load_dotenv()の後に初期化すること

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


@observe()
def get_embedding(text: str) -> list[float]:
    """Embeddingの生成をトレース"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


@observe()
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Vector DB検索をトレース"""
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    results = [
        {"title": r[0], "body": r[1], "similarity": round(r[2], 4)}
        for r in rows
    ]

    # v4: update_current_span()でメタデータを追加
    langfuse.update_current_span(
        metadata={
            "retrieved_count": len(results),
            "top_similarity": results[0]["similarity"] if results else 0,
        }
    )
    return results


@observe(name="llm_generate")
def generate_answer(question: str, context: str) -> str:
    """LLM回答生成をトレース"""
    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答（参考ドキュメントに基づいて簡潔に）"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


@observe(name="rag_pipeline")
def rag_answer(question: str) -> str:
    """
    RAGパイプライン全体をトレース
    Langfuseのダッシュボードで以下が確認できる:
    - rag_pipeline（全体）
      ├── search_documents（Vector DB検索）
      │   └── get_embedding（Embedding生成）
      └── llm_generate（LLM回答生成）
    """
    langfuse.update_current_span(
        metadata={"question": question, "tags": ["rag", "production"]}
    )

    docs = search_documents(question, top_k=3)
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])
    answer = generate_answer(question, context)
    return answer


if __name__ == "__main__":
    questions = [
        "F1スコアはどう計算しますか？",
        "AWSのコストを最適化する方法は？",
    ]

    for question in questions:
        print(f"\n質問: {question}")
        answer = rag_answer(question)
        print(f"回答: {answer[:100]}...")
        time.sleep(5)  # レート制限対策

    langfuse.flush()
    print("\nLangfuseにトレースを送信しました")
    print("https://cloud.langfuse.com でダッシュボードを確認してください")
```

```bash
mkdir observability
python observability/traced_rag.py
```

---

## Step 3: トレース付きAgent — `observability/traced_agent.py`

> **⚠️ ハマりやすいポイント2つ:**
> 1. `load_dotenv()` の後に `get_client()` を呼ぶ
> 2. `agent_step()` から `candidates` も返すこと（`run_agent()` 内で参照するため）

```python
# observability/traced_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langfuse import get_client, observe
import time

# ── 必ずload_dotenv()を先に呼ぶ ──────────────────────────────
load_dotenv()
langfuse = get_client()  # load_dotenv()の後に初期化すること

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


@observe(name="tool_search_documents")
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


@observe(name="tool_list_categories")
def list_categories() -> list[dict]:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="ドキュメントをVector DBから検索する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBのカテゴリ一覧を取得する。",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    return {"error": f"unknown function: {func_name}"}


@observe(name="agent_step")
def agent_step(contents: list, step_num: int) -> tuple:
    """
    Agentの1ステップをトレース

    Returns:
        tuple: (part, step_type, candidates)
        ※ candidatesもrun_agent()で参照するため返す必要がある
    """
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=[tools]),
            )
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 4:
                wait = (attempt + 1) * 10
                print(f"  リトライ {attempt+1}... {wait}秒待機")
                time.sleep(wait)
            else:
                raise

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return None, None, None

    part = candidates[0].content.parts[0]

    if part.function_call:
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        langfuse.update_current_span(
            metadata={"step": step_num, "tool": func_name, "args": str(func_args)}
        )
        return part, "tool_call", candidates
    else:
        langfuse.update_current_span(
            metadata={"step": step_num, "type": "final_answer"}
        )
        return part, "final", candidates


@observe(name="agent_pipeline")
def run_agent(task: str, max_steps: int = 5) -> str:
    """Agentパイプライン全体をトレース"""
    langfuse.update_current_span(
        metadata={"task": task, "tags": ["agent", "multi-step"]}
    )

    print(f"\nタスク: {task}")
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    step_count = 0

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")

        # candidatesもagent_step()から受け取る（run_agent内で参照するため）
        part, step_type, candidates = agent_step(contents, step + 1)

        if part is None:
            break

        if step_type == "tool_call":
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"  → {func_name}({func_args})")

            result = dispatch(func_name, func_args)
            print(f"  → {len(result) if isinstance(result, list) else result}件")

            contents.append(
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)])
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": result},
                        )
                    )]
                )
            )
            step_count += 1

        elif step_type == "final":
            text_parts = [
                p.text for p in candidates[0].content.parts
                if hasattr(p, 'text') and p.text
            ]
            answer = "\n".join(text_parts) if text_parts else ""

            langfuse.update_current_span(
                metadata={"total_steps": step_count + 1, "answer_length": len(answer)}
            )
            print(f"\n[完了] {step + 1}ステップで達成")
            return answer

    return "最大ステップ数に達しました。"


if __name__ == "__main__":
    result = run_agent(
        "まずカテゴリを確認して、MLカテゴリの評価指標について詳しく教えてください。"
    )
    print(f"\n最終回答:\n{result[:200]}...")

    langfuse.flush()
    print("\nLangfuseにトレースを送信しました")
    print("https://cloud.langfuse.com でダッシュボードを確認してください")
```

```bash
python observability/traced_agent.py
```

実行結果：

```
タスク: まずカテゴリを確認して、MLカテゴリの評価指標について詳しく教えてください。

[Step 1]
  → list_categories({})
  → 3件

[Step 2]
  → search_documents({'query': 'MLカテゴリの評価指標'})
  → 3件

[Step 3]
[完了] 3ステップで達成
最終回答:
MLカテゴリの評価指標についてですね。精度・再現率・F1スコアなどがあります...

Langfuseにトレースを送信しました
https://cloud.langfuse.com でダッシュボードを確認してください
```

---

## Step 4: ダッシュボードで確認できること

実行後に [cloud.langfuse.com](https://cloud.langfuse.com) を開くと以下が確認できます。

**Agentトレース（実際の表示）:**

| Name | Latency |
|------|---------|
| agent_pipeline | 4.40s |
| agent_step [1] | 1.34s（tool: list_categories） |
| tool_list_categories | 0.00s |
| agent_step [2] | 0.66s（tool: search_documents） |
| tool_search_documents | 0.42s |
| agent_step [3] | 1.97s（type: final_answer） |

---

## Langfuse v4の主な変更点（v3からの移行者向け）

| v3（旧） | v4（新） |
|---------|---------|
| `from langfuse.decorators import observe, langfuse_context` | `from langfuse import get_client, observe` |
| `from langfuse import Langfuse` → `Langfuse()` | `from langfuse import get_client` → `get_client()` |
| `langfuse_context.update_current_observation(...)` | `langfuse.update_current_span(...)` |
| `langfuse_context.update_current_trace(...)` | `langfuse.update_current_span(...)` |
| `update_current_observation` | `update_current_span` または `update_current_generation` |

---

## よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `Authentication error: initialized without public_key` | `get_client()` が `load_dotenv()` より前に呼ばれている | `load_dotenv()` の後に `get_client()` を呼ぶ |
| `cannot import name 'langfuse_context'` | v4では廃止 | `from langfuse import get_client, observe` に変更 |
| `has no attribute 'update_current_observation'` | v4では廃止 | `langfuse.update_current_span()` に変更 |
| `NameError: name 'candidates' is not defined` | `agent_step()` が `candidates` を返していない | `return part, step_type, candidates` に修正 |
| トレースが表示されない | `flush()` が呼ばれていない | スクリプト末尾に `langfuse.flush()` を追加 |
| `429 RESOURCE_EXHAUSTED` | Gemini無料枠の上限 | 翌日に再実行 |

---

## 次のステップ

- **[第4章 Security](./04_security)** — プロンプトインジェクション対策・ガードレール設計
- **Evalsとの統合** — LangfuseのスコアリングAPIでEvalsのスコアをトレースに紐付ける
- **継続的モニタリング** — 本番環境でのアラート設定

---

## 参考

- [Langfuse公式ドキュメント（v4）](https://langfuse.com/docs)
- [Langfuse Python v3→v4 移行ガイド](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4)
- [Observe デコレーター リファレンス](https://python.reference.langfuse.com/langfuse#observe)
