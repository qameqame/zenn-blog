---
title: "pgvectorの検索機能をMCPサーバーとして公開する"
---

## はじめに

[第5章](./05_agents)までで、Tool Useを使ってLLMが自律的にツールを呼び出す仕組みを実装しました。しかし`search_documents()` などのツール定義はPythonファイルの中に直書きしており、そのプロジェクトだけでしか使えない状態でした。

```
【Tool Use】
Pythonファイルの中に直書き → そのプロジェクトだけで使える

【今回 MCP】
独立したサーバーとして起動 → Claude Desktop・Gemini・どのLLMからも使える
```

**MCP（Model Context Protocol）** はAnthropicが策定したオープンなプロトコルで、ツール定義を標準化してどのLLMからでも同じ形式で使えるようにします。一度MCPサーバーを作れば、ClaudeでもGeminiでも同じツールを再利用できます。

---

## MCPの3つの概念

MCPには3つの基本要素があります。

| 要素 | 役割 | 今回の実装 |
|------|------|-----------|
| Tools | LLMが呼び出せる関数 | search_documents, search_by_category, list_categories |
| Resources | LLMが読めるデータ | DBのカテゴリ一覧（db://categories） |
| Prompts | 再利用可能なプロンプトテンプレート | 検索クエリのテンプレート |

---

## 前提条件

- 前回までのチュートリアル完了済み（pgvectorにドキュメントが格納済み）
- `.venv` が有効化済み
- `.env` が設定済み

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル（01〜11）
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py          # ★ MCPサーバー本体（今回追加）
│   └── client_test.py     # ★ サーバーのテスト（今回追加）
│
└── 12_mcp_agent.py        # ★ MCPサーバーを使うエージェント（今回追加）
```

| ファイル | 内容 |
|----------|------|
| `mcp_server/server.py` | MCPサーバー本体。`@mcp.tool` デコレーターでツールを登録 |
| `mcp_server/client_test.py` | サーバーに接続してツールをテストする |
| `12_mcp_agent.py` | MCPサーバーからツール定義を自動取得してエージェントを動かす |

---

## Tool Useとの違い

本記事で最も重要な変化は2点です。

**ツール定義の自動生成**

Tool Useでは `FunctionDeclaration` を手書きしていましたが、MCPでは `@mcp.tool` デコレーターと型ヒント・docstringだけでスキーマが自動生成されます。

**ツール実行の分離**

Tool Useでは `dispatch()` で直接関数を呼んでいましたが、MCPでは `mcp_client.call_tool()` 経由になります。サーバーが独立したプロセスになるため、Claude DesktopやGeminiなどどこからでも同じツールを使えます。

---

## 実装

### 1. ライブラリのインストール

```bash
pip install fastmcp
pip freeze > requirements.txt
```

### 2. MCPサーバーの実装 — `mcp_server/server.py`

```python
# mcp_server/server.py
import psycopg2
from google import genai
from google.genai import types as genai_types
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

# ── FastMCPサーバーの初期化 ──────────────────────────────────
# Tool Useでは client = genai.Client(...) だったが
# MCPでは mcp = FastMCP(...) がサーバーの起点になる
mcp = FastMCP(
    name="pgvector-search",
    instructions="pgvectorを使ったドキュメント検索サーバーです。"
                 "機械学習・Python・クラウドに関するドキュメントを検索できます。"
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


def get_embedding(text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


# ── Tool定義（@mcp.toolデコレーターで登録） ──────────────────
# Tool Useでは types.FunctionDeclaration(...) を手書きしていたが
# MCPでは @mcp.tool デコレーターと型ヒントだけで自動生成される

@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    全カテゴリのドキュメントからクエリに関連するものを検索する。
    カテゴリが不明な場合やカテゴリをまたぐ質問に使う。

    Args:
        query: 検索クエリ
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        タイトル・本文・カテゴリ・類似度スコアのリスト
    """
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


@mcp.tool
def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """
    特定カテゴリのドキュメントだけを検索する。
    ML・Python・Cloudなど明確にカテゴリが指定された質問に使う。

    Args:
        query: 検索クエリ
        category: カテゴリ名（ML / Python / Cloud）
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        タイトル・本文・カテゴリ・類似度スコアのリスト
    """
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


@mcp.tool
def list_categories() -> list[dict]:
    """
    DBに存在するカテゴリとドキュメント数の一覧を返す。
    どんなカテゴリがあるか確認したいときに使う。

    Returns:
        カテゴリ名とドキュメント数のリスト
    """
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


# ── Resource定義（読み取り専用データの公開） ─────────────────
# Resourceはツールではなく「LLMが読めるデータ」
# ドキュメントの一覧など静的な情報をResourceとして公開する

@mcp.resource("db://categories")
def get_categories_resource() -> str:
    """DBのカテゴリ一覧をResource（読み取り専用データ）として公開する"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    lines = [f"- {r[0]}: {r[1]}件" for r in rows]
    return "利用可能なカテゴリ:\n" + "\n".join(lines)


# ── Prompt定義（再利用可能なプロンプトテンプレート） ─────────
@mcp.prompt
def search_prompt(topic: str) -> str:
    """指定したトピックの検索プロンプトを生成する"""
    return f"""以下のトピックについてドキュメントを検索し、わかりやすくまとめてください。

トピック: {topic}

手順:
1. まず list_categories でカテゴリを確認する
2. 関連するカテゴリがあれば search_by_category で絞り込み検索
3. なければ search_documents で全体検索
4. 検索結果をもとに回答を生成する"""


# ── サーバーの起動 ────────────────────────────────────────────
if __name__ == "__main__":
    # stdioモード: Claude DesktopやMCPクライアントから起動される標準的な方法
    mcp.run()
```

> **Tool Useとの最大の違い:** Tool Useでは `types.FunctionDeclaration(name=..., description=..., parameters=...)` を手書きしていましたが、MCPでは `@mcp.tool` デコレーターとPythonの型ヒント・docstringだけでスキーマが自動生成されます。

### 3. `__init__.py` の作成

```bash
touch mcp_server/__init__.py
```

### 4. サーバーの動作テスト — `mcp_server/client_test.py`

```python
# mcp_server/client_test.py
import asyncio
from fastmcp import Client


async def test_server():
    """MCPサーバーのツールをPythonから直接テストする"""

    async with Client("mcp_server/server.py") as client:

        # 利用可能なツール一覧を確認
        tools = await client.list_tools()
        print("=== 利用可能なツール ===")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:40]}...")

        # Resourceの確認
        resources = await client.list_resources()
        print("\n=== 利用可能なリソース ===")
        for resource in resources:
            print(f"  - {resource.uri}")

        # ツールの実行テスト
        print("\n=== list_categories のテスト ===")
        result = await client.call_tool("list_categories", {})
        print(result)

        print("\n=== search_documents のテスト ===")
        result = await client.call_tool(
            "search_documents",
            {"query": "機械学習の評価指標", "top_k": 2}
        )
        print(result)

        print("\n=== search_by_category のテスト ===")
        result = await client.call_tool(
            "search_by_category",
            {"query": "モデル評価", "category": "ML", "top_k": 2}
        )
        print(result)

        # Resourceの読み取り
        print("\n=== Resource の読み取り ===")
        content = await client.read_resource("db://categories")
        print(content)


if __name__ == "__main__":
    asyncio.run(test_server())
```

```bash
python mcp_server/client_test.py
```

実行結果：

```
=== 利用可能なツール ===
  - search_documents: 全カテゴリのドキュメントからクエリに関連するもの...
  - search_by_category: 特定カテゴリのドキュメントだけを検索する...
  - list_categories: DBに存在するカテゴリとドキュメント数の一覧を返す...

=== 利用可能なリソース ===
  - db://categories

=== list_categories のテスト ===
[{'category': 'ML', 'count': 2}, {'category': 'Cloud', 'count': 2}, ...]

=== search_documents のテスト ===
[{'title': '機械学習モデルの評価指標', 'similarity': 0.8821}, ...]
```

### 5. MCPサーバーを使うエージェント — `12_mcp_agent.py`

今まで `dispatch()` を書いていた部分がMCPクライアント経由に変わります。

```python
# 12_mcp_agent.py
import asyncio
from google import genai
from google.genai import types
from fastmcp import Client
from dotenv import load_dotenv
import os
import time

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def run_agent(task: str):
    """MCPサーバーのツールを使うエージェント"""

    print(f"\nタスク: {task}")
    print("=" * 60)

    async with Client("mcp_server/server.py") as mcp_client:

        # MCPからツール定義を自動取得
        # Tool Useでは types.FunctionDeclaration を手書きしていたが
        # MCPではサーバーから自動でスキーマを取得できる
        mcp_tools = await mcp_client.list_tools()

        # MCPのツール定義をGemini用に変換する
        gemini_tools = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            param_name: types.Schema(
                                type=types.Type.STRING
                                if param_schema.get("type") == "string"
                                else types.Type.INTEGER
                                if param_schema.get("type") == "integer"
                                else types.Type.STRING,
                                description=param_schema.get("description", ""),
                            )
                            for param_name, param_schema in
                            (tool.inputSchema.get("properties") or {}).items()
                        },
                        required=tool.inputSchema.get("required", []),
                    ),
                )
                for tool in mcp_tools
            ]
        )

        print(f"MCPサーバーから{len(mcp_tools)}個のツールを取得しました")

        contents = [types.Content(role="user", parts=[types.Part(text=task)])]

        for step in range(8):
            print(f"\n[Step {step + 1}]")

            response = None
            for attempt in range(5):
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(tools=[gemini_tools]),
                    )
                    break
                except Exception as e:
                    if ("503" in str(e) or "429" in str(e)) and attempt < 4:
                        wait = (attempt + 1) * 10
                        print(f"  サーバー混雑、{wait}秒待ってリトライ...")
                        time.sleep(wait)
                    else:
                        raise

            candidates = response.candidates
            if not candidates or not candidates[0].content or not candidates[0].content.parts:
                return "（回答を取得できませんでした）"

            part = candidates[0].content.parts[0]

            if part.function_call:
                func_name = part.function_call.name
                func_args = dict(part.function_call.args)
                print(f"  → {func_name}({func_args})")

                # Tool Useでは dispatch() で直接関数を呼んでいたが
                # MCPではサーバー経由でツールを実行する
                result = await mcp_client.call_tool(func_name, func_args)
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
                text_parts = [
                    p.text for p in candidates[0].content.parts
                    if hasattr(p, 'text') and p.text
                ]
                print(f"\n[完了] {step + 1}ステップで達成")
                return "\n".join(text_parts)

    return "最大ステップ数に達しました。"


async def main():
    result = await run_agent(
        "まずカテゴリを確認して、MLカテゴリの評価指標について詳しく教えてください。"
    )
    print(f"\n最終回答:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python 12_mcp_agent.py
```

---

## Tool Useとの対比まとめ

| 項目 | Tool Use（06〜11） | MCP（今回） |
|------|-------------------|------------|
| ツール定義 | `FunctionDeclaration` を手書き | `@mcp.tool` デコレーターで自動生成 |
| ツール実行 | `dispatch()` 関数で直接呼ぶ | `mcp_client.call_tool()` 経由 |
| ツールの場所 | 同じPythonファイル内 | 独立したサーバープロセス |
| 再利用性 | プロジェクト内のみ | どのLLM・アプリからも使える |
| ツール一覧の取得 | コードで定義したものだけ | `list_tools()` でサーバーから動的取得 |

---

## Claude Desktopへの接続（オプション）

Claude Desktopがインストールされている場合、以下の設定で今回作ったサーバーをClaude Desktopから使えます。

`~/Library/Application Support/Claude/claude_desktop_config.json` に追加：

```json
{
  "mcpServers": {
    "pgvector-search": {
      "command": "python",
      "args": ["/Users/ユーザー名/XXXXX/pgvector-tutorial/mcp_server/server.py"],
      "env": {
        "GEMINI_API_KEY": "AIza...",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "vectordb",
        "DB_USER": "postgres",
        "DB_PASSWORD": "password"
      }
    }
  }
}
```

Claude Desktopを再起動すると、チャット画面から直接 `search_documents` などのツールが使えるようになります。

---

## 次のステップ

- **HTTPサーバー化** — `mcp.run(transport="streamable-http")` でリモートから使えるサーバーに
- **マルチエージェント** — 複数のMCPサーバーを組み合わせたエージェント設計
- **評価（Evals）** — ツールの回答品質を自動測定する仕組み

---

## 参考

- [第2章: RAG・Embedding・Vector DBの実装](./02_rag)
- [FastMCP ドキュメント](https://gofastmcp.com)
- [MCP 公式仕様](https://modelcontextprotocol.io)