---
title: "マルチエージェント — オーケストレーター × ワーカーパターンの実装"
---

## はじめに

[第6章のFine-tuning](./06_finetuning)まで、単一のAIシステムを改善することに集中してきました。本章では複数のAgentが協調する「マルチエージェント」設計を扱います。今まで実装してきたAgentは「1つのAgentが全部やる」設計でした。複雑なタスクになるほど、1つのAgentに全責任を負わせるのは限界があります。

```
【単一Agent（今まで）】
ユーザー → Agent → ツール → 回答
1つのAgentがすべてを判断・実行する

【マルチエージェント（今回）】
ユーザー → オーケストレーター → 検索Agent
                              → 回答生成Agent
                              → 品質チェックAgent
                           → 最終回答
役割を分担して協調する
```

マルチエージェントが有効な場面は3つあります。タスクが複数の専門性を必要とする場合、並列処理でスピードを上げたい場合、品質チェックを別のAgentに任せたい場合です。

> **2026年のトレンド:** Anthropicの研究ではマルチエージェントアーキテクチャでタスク完了率が90%改善。オーケストレーター × ワーカーパターンが本番での標準になっています。

---

## 本章で実装するアーキテクチャ

```
ユーザーの質問
    ↓
【オーケストレーター】（14_orchestrator.py）
  タスクを分析して2つのワーカーに依頼する
    ↓               ↓
【検索ワーカー】    【品質チェックワーカー】
pgvectorを検索     回答の品質を評価
    ↓               ↓
【オーケストレーター】
  結果を統合して最終回答を生成
    ↓
最終回答
```

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル（01〜13）
└── multiagent/
    ├── search_worker.py      # ★ 検索専門ワーカー
    ├── quality_worker.py     # ★ 品質チェック専門ワーカー
    ├── orchestrator.py       # ★ オーケストレーター
    └── 14_multiagent.py      # ★ 実行スクリプト
```

---

## 1. 検索ワーカー — `multiagent/search_worker.py`

pgvectorの検索だけに集中するワーカーです。単一の責任（検索）に特化することで、プロンプトがシンプルになり精度が上がります。

```python
# multiagent/search_worker.py
"""
検索専門ワーカー

役割: pgvectorからドキュメントを検索して返す
責任: 検索のみ（回答生成はしない）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

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

# 検索ワーカーのシステムプロンプト
# 単一の責任に集中したシンプルなプロンプト
SEARCH_WORKER_SYSTEM = """あなたはドキュメント検索の専門家です。
与えられた質問に関連するドキュメントをpgvectorから検索し、
検索結果をそのまま返してください。

制約:
- 回答を生成しない（検索結果を返すだけ）
- 検索結果がない場合は「関連ドキュメントが見つかりませんでした」と返す
- カテゴリが明確な場合はカテゴリ絞り込みを使う
"""


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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """全カテゴリから検索"""
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


def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """カテゴリ絞り込み検索"""
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE category = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, category, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


def list_categories() -> list[dict]:
    """カテゴリ一覧を取得"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


# ── Tool定義 ─────────────────────────────────────────────────
SEARCH_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="全カテゴリのドキュメントからクエリに関連するものを検索する",
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
            name="search_by_category",
            description="特定カテゴリのドキュメントだけを検索する",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING),
                    "category": types.Schema(type=types.Type.STRING, description="ML / Python / Cloud"),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                },
                required=["query", "category"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBのカテゴリ一覧を返す",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    return {"error": f"unknown: {func_name}"}


def run_search_worker(question: str) -> dict:
    """
    検索ワーカーのメイン関数

    Args:
        question: 検索する質問

    Returns:
        {
            "docs": [検索されたドキュメントのリスト],
            "steps": [実行したステップ数],
            "worker": "search"
        }
    """
    print(f"  [検索ワーカー] 質問: {question[:40]}...")

    contents = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    docs = []
    steps = 0

    for _ in range(5):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SEARCH_WORKER_SYSTEM,
                        tools=[SEARCH_TOOLS],
                    ),
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep((attempt + 1) * 10)
                else:
                    raise

        candidates = response.candidates
        if not candidates or not candidates[0].content.parts:
            break

        part = candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            result = dispatch(func_name, func_args)
            steps += 1
            print(f"  [検索ワーカー] {func_name} → {len(result) if isinstance(result, list) else result}件")

            # 検索結果を蓄積
            if func_name in ("search_documents", "search_by_category") and isinstance(result, list):
                docs.extend(result)

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
        else:
            break

    # 重複除去（タイトルで）
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc["title"] not in seen:
            seen.add(doc["title"])
            unique_docs.append(doc)

    print(f"  [検索ワーカー] 完了: {len(unique_docs)}件のドキュメントを取得")
    return {
        "docs": unique_docs,
        "steps": steps,
        "worker": "search",
    }
```

---

## 2. 品質チェックワーカー — `multiagent/quality_worker.py`

回答の品質を評価することだけに集中するワーカーです。

```python
# multiagent/quality_worker.py
"""
品質チェック専門ワーカー

役割: 生成された回答の品質を評価する
責任: 品質評価のみ（検索・回答生成はしない）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 品質チェックワーカーのシステムプロンプト
QUALITY_WORKER_SYSTEM = """あなたは回答品質の評価専門家です。
与えられた「質問・参照ドキュメント・回答」のセットを評価してください。

評価基準:
1. Faithfulness（忠実性）: 回答がドキュメントに基づいているか（0.0〜1.0）
2. Relevancy（関連性）: 質問に正しく答えているか（0.0〜1.0）
3. Completeness（完全性）: 必要な情報が含まれているか（0.0〜1.0）

必ず以下のJSON形式で返してください:
{
  "faithfulness": 0.0〜1.0,
  "relevancy": 0.0〜1.0,
  "completeness": 0.0〜1.0,
  "overall": 0.0〜1.0,
  "feedback": "改善点があれば記述"
}
"""


def run_quality_worker(question: str, docs: list[dict], answer: str) -> dict:
    """
    品質チェックワーカーのメイン関数

    Args:
        question: 元の質問
        docs: 検索されたドキュメント
        answer: 評価する回答

    Returns:
        {
            "faithfulness": float,
            "relevancy": float,
            "completeness": float,
            "overall": float,
            "feedback": str,
            "worker": "quality"
        }
    """
    print(f"  [品質ワーカー] 回答品質を評価中...")

    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    prompt = f"""以下を評価してください。

# 質問
{question}

# 参照ドキュメント
{context}

# 評価する回答
{answer}

JSON形式で評価結果を返してください。"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=QUALITY_WORKER_SYSTEM,
                ),
            )
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise

    import json
    import re
    raw = response.text.strip()
    # JSONブロックを抽出
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            result["worker"] = "quality"
            print(f"  [品質ワーカー] Overall: {result.get('overall', 'N/A')}")
            return result
        except json.JSONDecodeError:
            pass

    # パース失敗時のデフォルト
    print(f"  [品質ワーカー] 評価パース失敗、デフォルト値を使用")
    return {
        "faithfulness": 0.5,
        "relevancy": 0.5,
        "completeness": 0.5,
        "overall": 0.5,
        "feedback": "評価を取得できませんでした",
        "worker": "quality",
    }
```

---

## 3. オーケストレーター — `multiagent/orchestrator.py`

2つのワーカーを協調させる司令塔です。

```python
# multiagent/orchestrator.py
"""
オーケストレーター

役割: ユーザーの質問を受け取り、ワーカーに指示して結果を統合する
責任: タスク分解・ワーカー呼び出し・結果統合
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv
from multiagent.search_worker import run_search_worker
from multiagent.quality_worker import run_quality_worker
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# オーケストレーターのシステムプロンプト
ORCHESTRATOR_SYSTEM = """あなたはタスクを分解して複数のワーカーに指示するオーケストレーターです。

あなたの役割:
1. ユーザーの質問を受け取る
2. 検索ワーカーが取得したドキュメントをもとに回答を生成する
3. 品質チェックの結果を確認して必要なら回答を改善する
4. 最終回答をユーザーに返す

制約:
- 参照ドキュメントに基づいて回答すること
- 品質スコアが0.7未満の場合は回答を改善すること
- 簡潔で正確な回答を心がけること
"""


def generate_answer(question: str, docs: list[dict]) -> str:
    """検索結果をもとに回答を生成する"""
    time.sleep(15)  # レート制限対策（分間5リクエスト制限）
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参照ドキュメント
{context}

# 質問
{question}

# 回答（ドキュメントに基づいて簡潔に）"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ORCHESTRATOR_SYSTEM,
                ),
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise
    return "回答を生成できませんでした。"


def improve_answer(question: str, docs: list[dict], answer: str, feedback: str) -> str:
    """品質チェックのフィードバックをもとに回答を改善する"""
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    prompt = f"""以下の回答を改善してください。

# 質問
{question}

# 参照ドキュメント
{context}

# 現在の回答
{answer}

# 改善フィードバック
{feedback}

# 改善した回答"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise
    return answer


def run_orchestrator(question: str) -> dict:
    """
    オーケストレーターのメイン関数

    処理フロー:
    1. 検索ワーカーに検索を依頼
    2. 検索結果をもとに回答を生成
    3. 品質チェックワーカーに評価を依頼
    4. スコアが低ければ回答を改善
    5. 最終回答を返す

    Args:
        question: ユーザーの質問

    Returns:
        {
            "answer": 最終回答,
            "quality": 品質スコア,
            "docs_count": 検索したドキュメント数,
            "improved": 改善したかどうか
        }
    """
    print(f"\n{'='*60}")
    print(f"オーケストレーター起動")
    print(f"質問: {question}")
    print(f"{'='*60}")

    # ── Step 1: 検索ワーカーに検索を依頼 ────────────────────
    print("\n[Step 1] 検索ワーカーに依頼...")
    search_result = run_search_worker(question)
    docs = search_result["docs"]

    if not docs:
        return {
            "answer": "関連するドキュメントが見つかりませんでした。",
            "quality": {"overall": 0.0},
            "docs_count": 0,
            "improved": False,
        }

    print(f"  → {len(docs)}件のドキュメントを取得")

    # ── Step 2: 回答を生成 ───────────────────────────────────
    print("\n[Step 2] 回答を生成中...")
    answer = generate_answer(question, docs)
    print(f"  → 回答生成完了（{len(answer)}文字）")

    # ── Step 3: 品質チェックワーカーに評価を依頼 ────────────
    print("\n[Step 3] 品質チェックワーカーに依頼...")
    quality = run_quality_worker(question, docs, answer)

    improved = False

    # ── Step 4: 品質が低ければ改善 ──────────────────────────
    if quality.get("overall", 0) < 0.7:
        print(f"\n[Step 4] 品質スコア {quality.get('overall')} < 0.7 → 回答を改善...")
        feedback = quality.get("feedback", "回答を改善してください")
        answer = improve_answer(question, docs, answer, feedback)
        improved = True
        print(f"  → 回答改善完了")
    else:
        print(f"\n[Step 4] 品質スコア {quality.get('overall')} ≥ 0.7 → 改善不要")

    print(f"\n{'='*60}")
    print("オーケストレーター完了")
    print(f"{'='*60}")

    return {
        "answer": answer,
        "quality": quality,
        "docs_count": len(docs),
        "improved": improved,
    }
```

---

## 4. 実行スクリプト — `14_multiagent.py`

```python
# 14_multiagent.py
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent.orchestrator import run_orchestrator

def main():
    questions = [
        "MLカテゴリの評価指標について詳しく教えてください",
        "AWSのコスト削減方法は？",
    ]

    for question in questions:
        result = run_orchestrator(question)

        print(f"\n最終回答:")
        print(result["answer"])
        print(f"\n品質スコア:")
        q = result["quality"]
        print(f"  Faithfulness:  {q.get('faithfulness', 'N/A')}")
        print(f"  Relevancy:     {q.get('relevancy', 'N/A')}")
        print(f"  Completeness:  {q.get('completeness', 'N/A')}")
        print(f"  Overall:       {q.get('overall', 'N/A')}")
        print(f"  改善済み:      {'Yes' if result['improved'] else 'No'}")
        print(f"  使用ドキュメント: {result['docs_count']}件")
        print("\n" + "="*60)
        time.sleep(30)  # 質問間に30秒待機（レート制限対策）


if __name__ == "__main__":
    main()
```

---

## 5. 実行手順

```bash
mkdir multiagent
touch multiagent/__init__.py
python 14_multiagent.py
```

実行結果の例：

```
============================================================
オーケストレーター起動
質問: MLカテゴリの評価指標について詳しく教えてください
============================================================

[Step 1] 検索ワーカーに依頼...
  [検索ワーカー] 質問: MLカテゴリの評価指標について詳しく...
  [検索ワーカー] list_categories → 3件
  [検索ワーカー] search_by_category → 2件
  [検索ワーカー] 完了: 2件のドキュメントを取得

[Step 2] 回答を生成中...
  → 回答生成完了（324文字）

[Step 3] 品質チェックワーカーに依頼...
  [品質ワーカー] 回答品質を評価中...
  [品質ワーカー] Overall: 0.92

[Step 4] 品質スコア 0.92 ≥ 0.7 → 改善不要
============================================================
```

---

## 6. 設計パターンの比較

今回実装した「オーケストレーター × ワーカー」以外にも主要なパターンがあります。

| パターン | 構成 | 適した用途 |
|---------|------|----------|
| **オーケストレーター × ワーカー（今回）** | 1つの司令塔 + 複数の専門家 | 汎用タスク分解 |
| **シーケンシャルパイプライン** | Agent A → Agent B → Agent C | 文書処理・契約生成 |
| **並列Fan-out** | 複数Agentが同時実行 | 高速リサーチ・比較分析 |
| **階層型** | 上位オーケストレーター → 下位オーケストレーター → ワーカー | 大規模複雑タスク |

---

## 7. 本番運用での注意点

**コストの爆発に注意**
オーケストレーターが全ワーカーの結果を受け取ると、コンテキストウィンドウが肥大化します。ワーカーには必ず構造化JSON（要約）で返させ、全文を渡さないようにしましょう。

**ステップ数の上限を設ける**
暴走ループを防ぐため、各AgentにはStep上限（`max_steps`）を必ず設定します。

**エラーハンドリング**
1つのワーカーが失敗しても全体が止まらないよう、`try/except` でワーカーごとにエラーをハンドリングします。

---

## よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `ModuleNotFoundError: multiagent` | `__init__.py` がない | `touch multiagent/__init__.py` |
| `NameError: name 'time' is not defined` | `import time` 忘れ | `14_multiagent.py` の先頭に `import time` を追加 |
| `429 RESOURCE_EXHAUSTED` (分間制限) | Gemini無料枠5リクエスト/分 | `generate_answer()` の先頭に `time.sleep(15)` を追加 |
| `429 RESOURCE_EXHAUSTED` (日次制限) | Gemini無料枠20リクエスト/日 | 翌日に再実行 |
| 品質ワーカーのJSONパース失敗 | LLMが不正なJSON返す | `re.search` でJSONブロックを抽出 |
| コスト超過 | ワーカーへの全文送信 | 要約・構造化JSONで渡す |

---

## 次のステップ

- **[第8章 ガバナンス](./08_governance)** — マルチエージェントシステムのリスク管理・監査ログ設計
- **並列実行** — `asyncio.gather()` で検索ワーカーと品質チェックを並列化してスピードアップ
- **MCP × マルチエージェント** — ワーカーをMCPサーバーとして公開して再利用性を高める

---

## 参考

- [Anthropic マルチエージェント設計ガイド](https://docs.anthropic.com/en/docs/build-with-claude/agents)
- [LLM Workflow Patterns 2026](https://www.morphllm.com/llm-workflows)
- [Multi-Agent Orchestration Patterns](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production)
