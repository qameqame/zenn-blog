---
title: "Tool Use実装ガイド — LLMに検索を自律判断させる"
---

## はじめに

[第2章](./02_rag)では、pgvectorとGeminiを使ってRAGパイプラインを実装しました。そのシステムでは「毎回必ずsearchしてからLLMに渡す」という固定フローでした。

```
【前回 05_rag.py】固定フロー
質問 → 必ずsearch → LLMに渡す → 回答
```

今回実装する **Tool Use** を使うと、LLMが「この質問はDBを検索すべきか？」を自分で判断して、必要なツールを自律的に呼び出せるようになります。

```
【今回 Tool Use】
質問 → LLMが判断 → 必要なら search() を自分で呼ぶ → 回答
```

これがAI Agentsの第一歩です。

---

## 前提条件

- [第2章](./02_rag)のpgvectorチュートリアル完了済み（DBにドキュメントが格納されている）
- `.venv` が有効化済み
- `.env` が設定済み

---

## ディレクトリ構成

前回のファイルに3つ追加します。

```
pgvector-tutorial/
├── .env
├── .venv/
├── 01_setup_db.py       # 作成済み
├── 02_create_index.py   # 作成済み
├── 03_ingest.py         # 作成済み
├── 04_search.py         # 作成済み
├── 05_rag.py            # 作成済み
│
├── 06_tool_basic.py     # ★ Tool Useの基本（今回追加）
├── 07_tool_multi.py     # ★ 複数ツールの使い分け（今回追加）
└── 08_tool_agent.py     # ★ 自律的なループ処理（今回追加）
```

---

## Tool Useの全体像

### 3者の役割分担

Tool Useでは **LLM・Pythonコード・Vector DB** の3者が登場します。

```
【LLM】
・質問とツール定義を受け取る
・どのツールを使うか、引数に何を渡すかを判断する
・ツール名と引数をReturnする（関数は実行しない）
・最終的に検索結果を受け取って自然な文章で回答する

【Pythonコード】
・LLMからツール名と引数を受け取る
・実際に関数を実行する（dispatch）
・検索結果をLLMに返す

【Vector DB】
・ベクトル同士の距離を計算して類似ドキュメントを返す
・テキストは理解できない。数値（ベクトル）しか扱えない
```

### Tool Useの処理フロー

```
① 質問 + ツール定義 → LLMに送る

② LLMが返す（関数は実行しない、名前と引数を返すだけ）
   function_call.name = "search_by_category"
   function_call.args = {"query": "データサイエンス", "category": "ML"}

③ dispatch()がその名前と引数で実際の関数を呼ぶ
   search_by_category("データサイエンス", "ML")

④ search_by_category内でget_query_embedding()を呼ぶ
   "データサイエンス" → Geminiがベクトル化 → [0.12, -0.34, ...]

⑤ ベクトルでVector DBを検索
   SELECT * FROM documents WHERE category = 'ML'
   ORDER BY embedding <=> [0.12, -0.34, ...]

⑥ 検索結果（生データ）をLLMに返す

⑦ LLMが検索結果を自然な文章に変換して最終回答を生成
```

> **⑦最終回答が必要な理由:** Vector DBから返ってくるのはドキュメントの羅列です。LLMが「統合・解釈・生成」を行うことで、ユーザーが読める自然な回答になります。

### LLMはなぜ直接関数を実行しないのか

セキュリティ上の重要な設計です。LLMが勝手に意図しない関数を実行できないよう、**実行権限は常にPythonコード側** にあります。

```
LLMが返すもの = 「メモ書き」
  function_call.name = "search_by_category"
  function_call.args = {"query": "...", "category": "ML"}

Pythonコードが判断して実行
  → dispatch() が受け取って初めて search_by_category() を呼ぶ
```

---

## 実装

### 1. Tool Useの基本 — `06_tool_basic.py`

1つのツールを定義して、使うかどうかをLLMに判断させます。

```python
# 06_tool_basic.py
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


# ── 実際の関数（LLMがツール呼び出しを要求したら実行する） ──────────
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Vector DBからドキュメントを検索する"""

    # Step1: クエリをGeminiでベクトルに変換（Vector DBはテキストを直接扱えないため）
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    query_embedding = result.embeddings[0].values

    # Step2: ベクトルでVector DBを検索
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


# ── ツール定義（LLMに「こんな関数があります」と伝えるための定義） ──
# descriptionがLLMの判断基準になる。「いつ使うか」を具体的に書くほど精度が上がる
search_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="ユーザーの質問に答えるために必ず呼び出すツール。"
                        "自分の知識で答えられる場合でも、必ずこのツールで検索してから回答すること。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="検索クエリ。ユーザーの質問をそのまま渡す。",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="取得するドキュメント数。デフォルトは3。",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


# ── リトライ処理（503エラー対策） ─────────────────────────────────
def generate_with_retry(contents, tools_list, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools_list,
                    system_instruction="質問に答える前に必ず search_documents ツールを使って検索すること。",
                ),
            )
        except Exception as e:
            if "503" in str(e) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 5
                print(f"サーバー混雑、{wait}秒待ってリトライ...")
                time.sleep(wait)
            else:
                raise


# ── Tool Useのメイン処理 ────────────────────────────────────────
def ask_with_tool(question: str) -> str:
    print(f"\n質問: {question}")
    print("-" * 40)

    response = generate_with_retry(question, [search_tool])

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return "（LLMから回答がありませんでした）"

    # part.function_call がセットされていればツール呼び出し要求
    # part.function_call が None であればテキスト回答
    part = candidates[0].content.parts[0]

    if part.function_call:
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        print(f"LLMがツールを要求: {func_name}({func_args})")

        if func_name == "search_documents":
            search_result = search_documents(**func_args)
            print(f"検索結果: {len(search_result)}件取得")

        final_response = generate_with_retry(
            contents=[
                types.Content(role="user", parts=[types.Part(text=question)]),
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": search_result},
                        )
                    )]
                ),
            ],
            tools_list=[search_tool],
        )

        final_candidates = final_response.candidates
        if not final_candidates or not final_candidates[0].content or not final_candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        final_part = final_candidates[0].content.parts[0]
        if hasattr(final_part, 'text') and final_part.text:
            return final_part.text
        return "（回答を取得できませんでした）"

    else:
        print("LLMはツール不要と判断")
        return part.text


# ── 実行 ────────────────────────────────────────────────────────
print(ask_with_tool("F1スコアの計算方法を教えてください"))
print(ask_with_tool("今日は何曜日ですか？"))
```

```bash
python 06_tool_basic.py
```

実行結果：

```
質問: F1スコアの計算方法を教えてください
----------------------------------------
LLMがツールを要求: search_documents({'query': 'F1スコアの計算方法'})
検索結果: 3件取得
F1スコアはPrecisionとRecallの調和平均で計算します。
式は F1 = 2 × Precision × Recall ÷ (Precision + Recall) です。

質問: 今日は何曜日ですか？
----------------------------------------
LLMはツール不要と判断
申し訳ありませんが、私にはリアルタイムの日付情報へのアクセス手段がありません。
```

> **ポイント:** LLMは「F1スコア」はDBを検索すべき、「今日の曜日」はDBに関係ない、と自分で判断しています。

---

### 2. 複数ツールの使い分け — `07_tool_multi.py`

2つのツールを定義してLLMに使い分けさせます。

#### search_documentsとsearch_by_categoryの違い

どちらもVector DBを検索しますが、絞り込みの有無が違います。

```sql
-- search_documents: 全ドキュメントを対象に検索
SELECT title, body FROM documents
ORDER BY embedding <=> query_vec LIMIT 3;

-- search_by_category: カテゴリを絞ってから検索
SELECT title, body FROM documents
WHERE category = 'ML'        -- ここが違い
ORDER BY embedding <=> query_vec LIMIT 3;
```

#### ツールの選択はdescriptionで決まる

LLMは質問とツールの `description` を照合して選択します。`description` はLLMへの指示書です。「いつ使うか」を具体的に書くほど正確に選ばれます。

```
質問: 「ML分野でデータサイエンスを教えて」

search_documents の description:「複数カテゴリや不明な場合に使う」→ 該当しない
search_by_category の description:「1つのカテゴリに明確に絞った質問に使う」→ 該当する

→ search_by_category を選択
```

```python
# 07_tool_multi.py
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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """全カテゴリから検索"""
    query_embedding = get_query_embedding(query)
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
    """特定カテゴリに絞って検索"""
    query_embedding = get_query_embedding(query)
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


# ── ツール定義（2つ） ─────────────────────────────────────────
tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="複数カテゴリにまたがる質問や、カテゴリが不明な場合に全カテゴリから検索する。"
                        "「MLとCloudを比較」のように複数分野の質問は必ずこちらを使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数。指定がなければ3を使う。"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_by_category",
            description="ML・Python・Cloudのいずれか1つのカテゴリに明確に絞った質問のときだけ使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="カテゴリ名。ML / Python / Cloud のいずれか。",
                    ),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query", "category"],
            ),
        ),
    ]
)


# ── ツール呼び出しを実行するディスパッチャー ──────────────────────
# LLMは関数を直接実行できないため、Pythonコード側でこの処理が必要
def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    return {"error": f"unknown function: {func_name}"}


def generate_with_retry(contents, tools_list, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=tools_list),
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


def ask_with_tools(question: str) -> str:
    print(f"\n質問: {question}")
    print("-" * 40)

    response = generate_with_retry(question, [tools])

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return "（LLMから回答がありませんでした）"

    part = candidates[0].content.parts[0]

    if part.function_call:
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        print(f"LLMが選んだツール: {func_name}({func_args})")

        result = dispatch(func_name, func_args)
        print(f"結果: {len(result)}件取得")

        final_response = generate_with_retry(
            contents=[
                types.Content(role="user", parts=[types.Part(text=question)]),
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": result},
                        )
                    )]
                ),
            ],
            tools_list=[tools],
        )

        final_candidates = final_response.candidates
        if not final_candidates or not final_candidates[0].content or not final_candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        final_parts = final_candidates[0].content.parts
        text_parts = [p.text for p in final_parts if hasattr(p, 'text') and p.text]
        if text_parts:
            return "\n".join(text_parts)
        return "（複数ステップが必要な質問です → 08_tool_agent.py で試してください）"

    else:
        return part.text


# ── 実行 ────────────────────────────────────────────────────────
print(ask_with_tools("AWSのコスト最適化について教えてください"))
print(ask_with_tools("PythonでのデータサイエンスについてML分野から教えてください"))
```

```bash
python 07_tool_multi.py
```

---

### 3. 自律的なループ処理 — `08_tool_agent.py`

06・07は1回のツール呼び出しで終わります。08では「もう十分に答えられる」とLLMが判断するまでループします。

#### 2つの工夫

**工夫① 会話履歴の蓄積（contentsリスト）**

LLMは毎回のリクエストで記憶がリセットされます。`contents` にこれまでの会話と検索結果を全部詰め込んで渡すことで、LLMが「Step1で何を調べたか」を把握しながら次の判断ができます。

**工夫② ループ終了をLLMに判断させる**

「満足な回答ができるか」を明示的に判定するコードはありません。LLMがテキストで回答しようとした行為そのものが「満足」のシグナルになっています。

```python
if part.function_call:
    # オブジェクトがある = ツール呼び出し要求 = まだ情報が足りない → ループ継続
else:
    # None = テキスト回答 = 十分に答えられると判断した → ループ終了
    return part.text
```

```python
# 08_tool_agent.py
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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
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
    query_embedding = get_query_embedding(query)
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


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="全カテゴリのドキュメントから関連情報を検索する。",
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
            description="特定カテゴリ（ML・Python・Cloud）のドキュメントだけを検索する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "category": types.Schema(type=types.Type.STRING, description="カテゴリ名"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query", "category"],
            ),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
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


def agent(question: str, max_steps: int = 5) -> str:
    """
    LLMが自律的にツールを呼び出し続け、最終回答を出すまでループする。
    max_stepsは無限ループを防ぐための安全弁。
    """
    print(f"\n質問: {question}")
    print("=" * 50)

    # 工夫①: 会話履歴を蓄積するリスト
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")

        # 毎回全履歴を渡す（LLMが文脈を把握するため）
        response = generate_with_retry(contents, tools)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"ツール呼び出し: {func_name}({func_args})")

            result = dispatch(func_name, func_args)
            print(f"結果: {len(result)}件取得")

            # 工夫①: ツール要求と実行結果を履歴に追加して次のループへ
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
            # 工夫②: function_callがNone = テキスト回答 = ループ終了
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            print(f"\n最終回答（{step + 1}ステップで完了）:")
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ── 実行 ────────────────────────────────────────────────────────
answer = agent(
    "機械学習の評価方法を教えてください。"
    "まず全体を調べてから、MLカテゴリでも詳しく調べてください。"
)
print(answer)
```

```bash
python 08_tool_agent.py
```

実行結果：

```
質問: 機械学習の評価方法を教えてください。まず全体を調べてから、MLカテゴリでも詳しく調べてください。
==================================================
[Step 1]
ツール呼び出し: search_documents({'query': '機械学習の評価方法'})
結果: 3件取得

[Step 2]
ツール呼び出し: search_by_category({'query': '機械学習の評価方法', 'category': 'ML'})
結果: 3件取得

[Step 3]
最終回答（3ステップで完了）:
機械学習の評価方法には、精度・再現率・F1スコアなどの指標があります...
```

---

## 3ファイルの違いまとめ

| 項目 | 06 | 07 | 08 |
|------|----|----|-----|
| ツール数 | 1つ | 2つ | 2つ |
| ツール呼び出し回数 | 最大1回 | 最大1回 | 複数回（自律） |
| 会話履歴の蓄積 | なし | なし | あり（ループ） |
| LLMの自律度 | 低 | 中 | 高 |
| AI Agentsとの関係 | Tool Useの基本 | ツール選択 | Agentの第一歩 |

08が到達点です。LLMが「何を調べるか」「何回調べるか」「いつ終わるか」をすべて自分で判断します。

---

## 次のステップ

今回実装したTool Useを発展させると、以下に進めます。

- **AI Agents** — 検索・計算・API呼び出しなど複数種類のツールを組み合わせた本格的なエージェント設計
- **MCP（Model Context Protocol）** — ツール定義を標準化してどのLLMからでも使えるサーバーとして公開する
- **AIシステム設計** — Tool Useを組み込んだ本番システムのスケーリング設計

---

## 参考

- [第2章: pgvectorとGeminiで作るRAGパイプライン](./02_rag)
[第3章](./03_architect)
- [Google Gemini Function Calling ドキュメント](https://ai.google.dev/gemini-api/docs/function-calling)