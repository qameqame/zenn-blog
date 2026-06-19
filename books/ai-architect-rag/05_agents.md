---
title: "複数ツールを自律的に組み合わせるAI Agentsの実装"
---

## はじめに

[第4章](./04_tooluse)のTool Useでは、LLMが `search_documents` や `search_by_category` を自律的に呼び出す仕組みを実装しました。ただし、使えるツールは「検索」の1種類だけでした。

今回は **AI Agents** に進みます。検索・計算・メモリなど **複数種類のツールを組み合わせて**、より複雑なタスクを自律的に達成するエージェントを実装します。

```
【前回 Tool Use】
ツール: search_documents, search_by_category のみ

【今回 AI Agents】
ツール: 検索 + カテゴリ取得 + 統計計算 + メモリ を組み合わせる
```

AI Agentsとは「人間が手順を指示しなくても、LLMが自分で計画・実行・修正してタスクを達成する」システムです。

---

## 前提条件

- 前回のTool Useチュートリアル完了済み（`08_tool_agent.py` が動作済み）
- `.venv` が有効化済み
- `.env` が設定済み

---

## ディレクトリ構成

前回のファイルに3つ追加します。

```
pgvector-tutorial/
├── 既存ファイル（01〜08）
│
├── 09_agent_basic.py      # ★ 複数ツールを持つエージェント（今回追加）
├── 10_agent_memory.py     # ★ メモリ機能付きエージェント（今回追加）
└── 11_agent_planner.py    # ★ 計画→実行→評価のループ（今回追加）
```

---

## AI Agentsの設計原則

### ReActパターン（Reasoning + Acting）

AI Agentsの最も基本的な設計パターンです。

```
Reasoning（考える）: 「次に何をすべきか」を推論する
Acting（行動する）  : ツールを呼び出して情報を得る
Observation（観察）: 結果を見て次の推論に活かす

→ このループを繰り返して最終目標に到達する
```

前回実装した08のAgenticループはまさにこのパターンです。

```python
for step in range(max_steps):
    response = llm(contents)      # Reasoning: 次に何をすべきか考える

    if part.function_call:
        result = dispatch(...)    # Acting: ツールを実行する
        contents.append(result)   # Observation: 結果を履歴に追加
    else:
        return part.text          # 目標達成 → ループ終了
```

### Tool Useとの違い

| 項目 | Tool Use（08） | AI Agents（今回） |
|------|--------------|-----------------|
| ツールの種類 | 検索のみ | 検索・計算・メモリなど複数 |
| タスクの複雑さ | 単純な情報検索 | 複数ステップの複合タスク |
| エラー処理 | 最小限 | リトライ・代替手段の選択 |
| メモリ | 会話履歴のみ | 外部への永続化も可能 |

---

## 実装

### 1. 複数ツールを持つエージェント — `09_agent_basic.py`

検索・カテゴリ取得・統計計算の3種類のツールを組み合わせます。

```python
# 09_agent_basic.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import json

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


# ── ツール①: Vector DB検索 ────────────────────────────────────
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Vector DBからドキュメントを検索する"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    query_embedding = result.embeddings[0].values
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


# ── ツール②: カテゴリ一覧取得 ────────────────────────────────
def list_categories() -> list[dict]:
    """DBに存在するカテゴリとドキュメント数を返す"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


# ── ツール③: 類似度スコアの統計計算 ─────────────────────────
def calculate_stats(scores: list[float]) -> dict:
    """類似度スコアの統計情報を計算する"""
    if not scores:
        return {"error": "スコアが空です"}
    return {
        "count": len(scores),
        "average": round(sum(scores) / len(scores), 4),
        "max": round(max(scores), 4),
        "min": round(min(scores), 4),
    }


# ── ツール定義 ──────────────────────────────────────────────
tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="ユーザーの質問に関連するドキュメントをVector DBから検索する。"
                        "情報を調べる必要があるときに使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数。デフォルト3。"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBに存在するカテゴリとドキュメント数の一覧を取得する。"
                        "どんなカテゴリがあるか確認したいときに使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="calculate_stats",
            description="類似度スコアのリストを受け取り、平均・最大・最小などの統計情報を計算する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "scores": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.NUMBER),
                        description="類似度スコアのリスト（0〜1の数値）",
                    ),
                },
                required=["scores"],
            ),
        ),
    ]
)


# ── ディスパッチャー ─────────────────────────────────────────
def dispatch(func_name: str, func_args: dict):
    """LLMから返ってきたツール名で実際の関数を呼び出す"""
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    elif func_name == "calculate_stats":
        return calculate_stats(**func_args)
    return {"error": f"unknown function: {func_name}"}


def generate_with_retry(contents, tools_obj, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=[tools_obj]),
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


# ── Agentのメインループ（ReActパターン） ─────────────────────
def agent(task: str, max_steps: int = 8) -> str:
    print(f"\nタスク: {task}")
    print("=" * 60)

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}] Reasoning...")

        response = generate_with_retry(contents, tools)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
            # Acting: ツールを実行する
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"  → Acting: {func_name}({func_args})")

            result = dispatch(func_name, func_args)

            # Observation: 結果を表示して履歴に追加
            if isinstance(result, list):
                print(f"  → Observation: {len(result)}件取得")
            else:
                print(f"  → Observation: {result}")

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
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            print(f"\n[完了] {step + 1}ステップで達成")
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ── 実行 ────────────────────────────────────────────────────
# タスク①: 複数ツールの連携（カテゴリ確認 → 検索）
print(agent(
    "まずDBにどんなカテゴリがあるか調べて、"
    "その後でPythonカテゴリのドキュメントを検索してください。"
))

# タスク②: 3ツールの連携（検索 → 統計計算）
print(agent(
    "機械学習の評価指標についてのドキュメントを検索して、"
    "見つかったドキュメントの類似度スコアの統計情報も計算してください。"
))
```

```bash
python 09_agent_basic.py
```

実行結果：

```
タスク: まずDBにどんなカテゴリがあるか調べて...
============================================================
[Step 1] Reasoning...
  → Acting: list_categories({})
  → Observation: 3件取得

[Step 2] Reasoning...
  → Acting: search_documents({'query': 'Python', 'top_k': 3})
  → Observation: 3件取得

[完了] 2ステップで達成
DBには ML・Python・Cloud の3カテゴリがあります。
Pythonカテゴリには「Pandasによるデータ前処理」が含まれています...
```

> **ポイント:** LLMが「まずカテゴリを調べる → 次に検索する」という順序を自分で計画して実行しています。この計画能力がAI Agentsの核心です。

---

### 2. メモリ機能付きエージェント — `10_agent_memory.py`

会話をまたいで記憶を持つエージェントを実装します。AgentのメモリはShort-term（会話履歴）とLong-term（外部保存）の2種類があります。

```python
# 10_agent_memory.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime

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


# ── Long-term Memory: ファイルで永続化 ──────────────────────
MEMORY_FILE = "agent_memory.json"

def load_memory() -> dict:
    """保存済みの記憶を読み込む"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"facts": [], "searches": []}

def save_memory(memory: dict):
    """記憶をファイルに保存する"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


# ── ツール定義（検索 + メモリ操作） ─────────────────────────
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    query_embedding = result.embeddings[0].values
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

def remember_fact(fact: str) -> dict:
    """重要な情報を長期記憶に保存する"""
    memory = load_memory()
    memory["facts"].append({
        "fact": fact,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)
    return {"status": "保存しました", "fact": fact}

def recall_facts() -> list[dict]:
    """長期記憶から保存済みの情報を取り出す"""
    memory = load_memory()
    return memory["facts"]

def clear_memory() -> dict:
    """長期記憶をリセットする"""
    save_memory({"facts": [], "searches": []})
    return {"status": "メモリをクリアしました"}


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Vector DBからドキュメントを検索する。",
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
            name="remember_fact",
            description="重要な情報や学んだことを長期記憶に保存する。次回の会話でも使いたい情報はここに保存する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "fact": types.Schema(type=types.Type.STRING, description="保存する情報"),
                },
                required=["fact"],
            ),
        ),
        types.FunctionDeclaration(
            name="recall_facts",
            description="長期記憶に保存されている情報を取り出す。以前学んだことを確認したいときに使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="clear_memory",
            description="長期記憶をすべて削除してリセットする。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "remember_fact":
        return remember_fact(**func_args)
    elif func_name == "recall_facts":
        return recall_facts()
    elif func_name == "clear_memory":
        return clear_memory()
    return {"error": f"unknown function: {func_name}"}


def generate_with_retry(contents, tools_obj, system_instruction=None, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            config_args = {"tools": [tools_obj]}
            if system_instruction:
                config_args["system_instruction"] = system_instruction
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(**config_args),
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


def agent(task: str, max_steps: int = 8) -> str:
    print(f"\nタスク: {task}")
    print("=" * 60)

    # システムプロンプトでエージェントの行動指針を定義
    system_instruction = """あなたは学習支援エージェントです。
- 重要な情報を学んだら remember_fact で長期記憶に保存してください
- タスク開始時に recall_facts で過去の記憶を確認してください
- 記憶を活用して、より質の高い回答を提供してください"""

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}] Reasoning...")

        response = generate_with_retry(contents, tools, system_instruction)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"  → {func_name}({func_args})")

            result = dispatch(func_name, func_args)
            print(f"  → {len(result) if isinstance(result, list) else result}")

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
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            print(f"\n[完了] {step + 1}ステップで達成")
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ── 実行 ────────────────────────────────────────────────────
# 1回目: 学習して記憶に保存
print(agent("F1スコアについて調べて、重要なポイントを長期記憶に保存してください。"))

# 2回目: 記憶を使って回答（DBを再検索しなくても答えられるはず）
print(agent("以前学んだF1スコアの情報を思い出して、簡単にまとめてください。"))
```

```bash
python 10_agent_memory.py
```

> **ポイント:** 1回目の実行で `agent_memory.json` が作成されます。2回目は `recall_facts` でファイルから記憶を読み出し、DBを再検索しなくても答えられます。これがLong-term Memoryの基本パターンです。

---

### 3. 計画→実行→評価のループ — `11_agent_planner.py`

より高度なエージェントは「まず計画を立ててから実行する」設計を使います。

```python
# 11_agent_planner.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
    )
    query_embedding = result.embeddings[0].values
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [{"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)} for r in rows]


def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
    )
    query_embedding = result.embeddings[0].values
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents WHERE category = %s
        ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (query_embedding, category, query_embedding, top_k))
    rows = cur.fetchall()
    return [{"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)} for r in rows]


def list_categories() -> list[dict]:
    cur.execute("SELECT category, COUNT(*) as count FROM documents GROUP BY category ORDER BY count DESC;")
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="全カテゴリから関連ドキュメントを検索する。",
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
            description="特定カテゴリのドキュメントだけを検索する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "category": types.Schema(type=types.Type.STRING, description="カテゴリ名（ML/Python/Cloud）"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query", "category"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBに存在するカテゴリ一覧を取得する。",
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
    return {"error": f"unknown function: {func_name}"}


def generate_with_retry(contents, tools_obj, system_instruction=None, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            config_args = {"tools": [tools_obj]}
            if system_instruction:
                config_args["system_instruction"] = system_instruction
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(**config_args),
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


# ── Phase 1: 計画（Plan） ────────────────────────────────────
# LLMにまず「何をどの順番でやるか」を考えさせる
def plan(task: str) -> str:
    print(f"\n[Phase 1] 計画を立てています...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""以下のタスクを達成するための実行計画を立ててください。
使えるツール: search_documents, search_by_category, list_categories

タスク: {task}

計画を箇条書きで簡潔に書いてください（3〜5ステップ）。""",
    )
    plan_text = response.candidates[0].content.parts[0].text
    print(f"計画:\n{plan_text}")
    return plan_text


# ── Phase 2: 実行（Execute） ─────────────────────────────────
# 計画に基づいてエージェントを実行する
def execute(task: str, plan_text: str, max_steps: int = 8) -> str:
    print(f"\n[Phase 2] 計画を実行しています...")

    system_instruction = f"""以下の計画に従ってタスクを実行してください。

計画:
{plan_text}

計画通りに進め、各ステップでツールを使って情報を収集してから最終回答を生成してください。"""

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")
        response = generate_with_retry(contents, tools, system_instruction)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
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
        else:
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ── Phase 3: 評価（Evaluate） ────────────────────────────────
# 結果がタスクを満たしているか評価する
def evaluate(task: str, result: str) -> str:
    print(f"\n[Phase 3] 結果を評価しています...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""以下のタスクと回答を評価してください。

タスク: {task}
回答: {result}

評価項目:
1. タスクの要求を満たしているか（Yes/No）
2. 不足している情報があれば指摘してください
3. 総合評価（1〜5点）

簡潔に回答してください。""",
    )
    evaluation = response.candidates[0].content.parts[0].text
    print(f"評価結果:\n{evaluation}")
    return evaluation


# ── Plan → Execute → Evaluate の統合実行 ────────────────────
def plan_execute_evaluate(task: str):
    print(f"\n{'='*60}")
    print(f"タスク: {task}")
    print(f"{'='*60}")

    plan_text = plan(task)
    result = execute(task, plan_text)
    print(f"\n[実行結果]\n{result}")
    evaluate(task, result)


# ── 実行 ────────────────────────────────────────────────────
plan_execute_evaluate(
    "機械学習とクラウド技術の両方について調査して、"
    "それぞれの主要なトピックをまとめてください。"
)
```

```bash
python 11_agent_planner.py
```

> **ポイント:** Plan→Execute→Evaluate の3フェーズに分けることで「何をやるか決める」「実際にやる」「結果を検証する」が分離されます。本番システムではEvaluateの結果が不十分なら再実行する設計になります。

---

## 3ファイルで学んだAI Agentsのパターン

| ファイル | パターン | 学んだこと |
|----------|----------|-----------|
| `09_agent_basic.py` | ReActループ | 複数ツールの連携・LLMによる計画能力 |
| `10_agent_memory.py` | Memory-augmented | Short/Long-term Memory・ファイルへの永続化 |
| `11_agent_planner.py` | Plan-Execute-Evaluate | 計画→実行→評価の3フェーズ設計 |

---

## AIアーキテクトとして押さえるべき設計ポイント

**ツールの粒度設計**

ツールは「1つのことだけをする」単機能にします。`search_and_summarize()` のような複合ツールより、`search()` と `summarize()` に分けた方がLLMが使いやすくなります。

**max_stepsの設定**

無限ループ防止のための重要な設定です。タスクの複雑さに応じて調整します。シンプルなタスクは3〜5、複雑なタスクは8〜10が目安です。

**システムプロンプトの役割**

エージェントの「役割」「行動指針」「制約」をシステムプロンプトで定義します。同じツールセットでも、システムプロンプトの違いで全く異なる動作になります。

---

## 次のステップ

- **MCP（Model Context Protocol）** — ツール定義を標準化して外部サービスと簡単に繋げる
- **マルチエージェント** — 複数のエージェントが協調して1つのタスクを分担する設計
- **AIシステム設計** — AgentをAPIとして本番公開するスケーリング設計
- **評価（Evals）** — Agentの回答品質を自動測定する仕組み

---

## 参考

- [第2章: RAG・Embedding・Vector DBの実装](./02_rag)
- [Google Gemini Function Calling ドキュメント](https://ai.google.dev/gemini-api/docs/function-calling)