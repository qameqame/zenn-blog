---
title: "MCPサーバーをクラウドに公開する — Render × Supabase 実践"
---

## はじめに

[第6章](./06_mcp)では、pgvectorの検索機能をMCPサーバーとして公開し、Claude Desktopからローカルで使えるようになりました。しかしこの構成はあくまでローカル環境限定です。

```
【第6章まで】ローカル環境
Claude Desktop → localhost:8000 → pgvector（Docker）

【今章】クラウド環境
Python Agent → https://your-app.onrender.com/mcp → Supabase（pgvector）
```

本章では、MCPサーバーをRenderにデプロイし、pgvectorデータベースをSupabaseに移行します。どちらもクレジットカード不要の無料枠で動かせます。

> **Claude DesktopとHTTPモードについて:**
> 現時点でClaude DesktopはMCPのHTTPトランスポートに対応していません。
> クラウドデプロイしたMCPサーバーはPythonスクリプト（`13_mcp_http_agent.py`）から接続します。
> Claude Desktopからは引き続きstdioモード（`server.py`）を使います。

| サービス | 役割 | 無料枠 |
|----------|------|--------|
| **Render** | MCPサーバーのホスティング | Webサービス永続無料（15分でスリープ） |
| **Supabase** | pgvectorデータベース | 500MB・永続（7日無操作でポーズ） |

---

## 全体の手順

```
Step 1: Supabaseでpgvectorを設定
Step 2: ローカルのデータをSupabaseに移行
Step 3: MCPサーバーをRender用に修正
Step 4: GitHubにプッシュ
Step 5: Renderにデプロイ
Step 6: Pythonエージェントから接続確認
```

---

## Step 1: Supabaseのセットアップ

### 1-1. プロジェクト作成

1. [supabase.com](https://supabase.com) にアクセス → GitHubアカウントでサインアップ
2. 「New project」をクリック
3. 以下を設定：
   - **Name:** `pgvector-tutorial`
   - **Database Password:** 任意のパスワード（後で使うので必ず控えておく）
   - **Region:** `Northeast Asia (Tokyo)`
4. 「Create new project」をクリック（2〜3分かかります）

### 1-2. pgvector拡張を有効化

左サイドバーの **`>_`（SQL Editor）** を開いて以下を実行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1-3. テーブルとインデックスを作成

引き続きSQL Editorで実行：

```sql
-- テーブル作成
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    category    TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    embedding   vector(768)
);

-- HNSWインデックス作成
CREATE INDEX IF NOT EXISTS docs_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- カテゴリインデックス
CREATE INDEX ON documents (category);
```

### 1-4. 接続情報を取得（重要: Connection Poolerを使う）

> **⚠️ ここでハマりやすいポイント:**
> 通常のDB接続URL（ポート5432）を使うと、RenderからSupabaseへの接続がIPv6問題で失敗します。
> 必ず **Connection Pooler（ポート6543）** のURLを使ってください。

画面上部の **「Connect」** ボタンをクリックします。

> `Settings` → `Database` の中には接続情報が見つかりにくい場合があります。
> 画面上部の緑色の「Connect」ボタンが確実です。

「Connect」ダイアログで **「Connection pooling」** タブを選択し、**「Transaction」** モードのURIをコピーします。

URLはこのような形式です：

```
postgresql://postgres.xxxx:パスワード@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres
```

| 比較 | 通常接続 | Connection Pooler |
|------|---------|-------------------|
| ポート | 5432 | **6543** |
| ホスト | `db.xxxx.supabase.co` | `aws-0-xxx.pooler.supabase.com` |
| IPv6問題 | ❌ Renderで失敗する | ✅ IPv4で安定動作 |
| Render向け | ❌ | ✅ |

---

## Step 2: ローカルのデータをSupabaseに移行

### 2-1. `.env` にSupabaseの接続情報を追加

```
# Gemini API
GEMINI_API_KEY=AIza...

# ローカルDB（従来）
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password

# Supabase（Connection Pooler URLを使う）
DATABASE_URL=postgresql://postgres.xxxx:パスワード@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres
```

### 2-2. データ移行スクリプト — `migrate_to_supabase.py`

```python
# migrate_to_supabase.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# ── ローカルDBに接続 ──────────────────────────────────────────
local_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
local_cur = local_conn.cursor()

# ── Supabaseに接続（Connection Pooler経由） ───────────────────
supa_conn = psycopg2.connect(
    os.getenv("DATABASE_URL"),
    sslmode="require",  # Supabaseは SSL必須
)
supa_cur = supa_conn.cursor()

# ── ローカルのデータを取得 ────────────────────────────────────
local_cur.execute("SELECT title, body, category, embedding FROM documents;")
rows = local_cur.fetchall()
print(f"移行するドキュメント数: {len(rows)}")

# ── Supabaseに挿入 ────────────────────────────────────────────
for row in rows:
    title, body, category, embedding = row
    supa_cur.execute("""
        INSERT INTO documents (title, body, category, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
    """, (title, body, category, embedding))

supa_conn.commit()
print("移行完了！")

supa_cur.execute("SELECT COUNT(*) FROM documents;")
count = supa_cur.fetchone()[0]
print(f"Supabase内のドキュメント数: {count}")

local_conn.close()
supa_conn.close()
```

```bash
python migrate_to_supabase.py
# => 移行するドキュメント数: 5
# => 移行完了！
# => Supabase内のドキュメント数: 5
```

---

## Step 3: MCPサーバーをRender用に修正

`mcp_server/server_render.py` を作成します。`server_http.py` との違いは DB接続部分と `PORT` 環境変数の対応だけです。

```python
# mcp_server/server_render.py
import psycopg2
from google import genai
from google.genai import types as genai_types
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP(
    name="pgvector-search",
    instructions="pgvectorを使ったドキュメント検索サーバーです。"
                 "機械学習・Python・クラウドに関するドキュメントを検索できます。"
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── DB接続（DATABASE_URL環境変数を使う） ──────────────────────
# RenderのダッシュボードでDATABASE_URLを設定する
# 必ずSupabaseのConnection Pooler URL（ポート6543）を使うこと
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
else:
    # ローカルテスト用フォールバック
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "vectordb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
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


@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    全カテゴリのドキュメントからクエリに関連するものを検索する。

    Args:
        query: 検索クエリ
        top_k: 取得するドキュメント数（デフォルト: 3）
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

    Args:
        query: 検索クエリ
        category: カテゴリ名（ML / Python / Cloud）
        top_k: 取得するドキュメント数（デフォルト: 3）
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
    """DBに存在するカテゴリとドキュメント数の一覧を返す。"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


@mcp.resource("db://categories")
def get_categories_resource() -> str:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    lines = [f"- {r[0]}: {r[1]}件" for r in rows]
    return "利用可能なカテゴリ:\n" + "\n".join(lines)


if __name__ == "__main__":
    # RenderはPORT環境変数を自動設定する
    port = int(os.getenv("PORT", 8000))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )
```

---

## Step 4: GitHubにプッシュ

```bash
git add .
git commit -m "feat: add Render deployment server

- Add server_render.py with Connection Pooler support
- Add migrate_to_supabase.py for data migration"

git push origin main
```

---

## Step 5: Renderにデプロイ

### 5-1. Renderアカウント作成

1. [render.com](https://render.com) → 「Get Started for Free」→ GitHubでサインアップ
2. クレジットカード不要

### 5-2. Webサービスを作成

1. ダッシュボードの「New」→「Web Service」
2. GitHubリポジトリを選択
3. 以下を設定：

| 項目 | 値 |
|------|-----|
| Name | `pgvector-mcp-server` |
| Region | `Oregon (US West)` |
| Branch | `main` |
| Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python mcp_server/server_render.py` |
| Instance Type | `Free` |

4. 「Create Web Service」をクリック

### 5-3. 環境変数を設定

「Environment」タブで以下を追加：

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | `AIza...` |
| `DATABASE_URL` | **Connection PoolerのURI（ポート6543）** |

> **⚠️ DATABASE_URLは必ずConnection Pooler（ポート6543）のURLを使ってください。**
> 通常のDB URL（ポート5432）を使うとRenderからの接続がIPv6問題で失敗します。

「Save Changes」でデプロイが自動開始されます。

### 5-4. デプロイ確認

ブラウザで以下にアクセスして応答があれば成功です：

```
https://pgvector-mcp-server.onrender.com/mcp
```

> **「Not Acceptable: Client must accept text/event-stream」** と表示されれば正常です。
> これはブラウザ（MCPクライアント以外）からアクセスしているために出るメッセージで、サーバーは正常に動作しています。

---

## Step 6: Pythonエージェントから接続確認

`13_mcp_http_agent.py` のURLをRenderのURLに変更します：

```python
# 変更前
MCP_SERVER_URL = "http://localhost:8000/mcp"

# 変更後
MCP_SERVER_URL = "https://pgvector-mcp-server.onrender.com/mcp"
```

```bash
python 13_mcp_http_agent.py
```

実行結果：

```
タスク: まずカテゴリを確認して、MLカテゴリの評価指標について詳しく教えてください。
============================================================
MCPサーバーから3個のツールを取得しました（HTTP経由）

[Step 1]
  → list_categories({})

[Step 2]
  → search_by_category({'query': '評価指標', 'category': 'ML'})

[完了] 3ステップで達成
MLカテゴリの評価指標についてですね...
```

`（HTTP経由）` の表示がRenderのMCPサーバーを経由していることの証明です。

---

## スリープ対策（オプション）

Renderの無料サービスは15分間アクセスがないとスリープします。[UptimeRobot](https://uptimerobot.com)（無料）で5分ごとにpingを送ると常時起動状態を維持できます。

1. UptimeRobotに登録
2. 「Add New Monitor」→ HTTP(s)
3. URL: `https://pgvector-mcp-server.onrender.com/mcp`
4. Monitoring Interval: `5 minutes`

---

## よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `Network is unreachable` (IPv6) | 通常のDB URL（ポート5432）を使っている | **Connection Pooler URL（ポート6543）に変更** |
| `SSL connection required` | sslmode未指定 | `sslmode="require"` を追加 |
| `ModuleNotFoundError` | requirements.txtに不足 | `pip freeze > requirements.txt` して再push |
| `Not Found` | ルートURL（`/`）にアクセスしている | `/mcp` を付けてアクセス |
| `Not Acceptable: Client must accept text/event-stream` | ブラウザでアクセスしている | 正常。MCPクライアントからは接続できる |
| デプロイ後に接続が遅い | Renderのスリープ | UptimeRobotで定期ping |
| Supabaseがポーズ状態 | 7日間アクセスなし | Supabaseダッシュボードから「Resume」 |

---

## 参考

- [Render公式ドキュメント](https://render.com/docs)
- [Supabase pgvectorドキュメント](https://supabase.com/docs/guides/database/extensions/pgvector)
- [FastMCP HTTPトランスポート](https://gofastmcp.com/deployment/running-server)
