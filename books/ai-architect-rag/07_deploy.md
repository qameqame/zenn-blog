---
title: "クラウドデプロイ — Render × Supabase で本番環境を作る"
---

## MCPサーバーをクラウドにデプロイする


---

## はじめに

ローカルで動いていたMCPサーバーをインターネット上に公開します。

```
【今まで】ローカル環境
Claude Desktop → localhost:8000 → pgvector（Docker）

【今回】クラウド環境
Claude Desktop → https://your-app.onrender.com/mcp → Supabase（pgvector）
```

使うサービスと無料枠：

| サービス | 役割 | 無料枠 |
|----------|------|--------|
| **Render** | MCPサーバーのホスティング | Webサービス永続無料（スリープあり） |
| **Supabase** | pgvectorデータベース | 500MB・永続（7日無操作でポーズ） |

どちらもクレジットカード不要です。

> **注意点:**
> - Renderの無料Webサービスは15分間アクセスがないとスリープします（再起動30〜60秒）
> - Supabaseの無料プロジェクトは7日間APIアクセスがないとポーズします（手動で再開可能）
> - 学習・デモ目的には十分です

---

## 全体の手順

```
Step 1: Supabaseでpgvectorを設定
Step 2: ローカルのデータをSupabaseに移行
Step 3: MCPサーバーをRender用に修正
Step 4: GitHubにプッシュ
Step 5: Renderにデプロイ
Step 6: Claude Desktopから接続確認
```

---

## Step 1: Supabaseのセットアップ

### 1-1. プロジェクト作成

1. [supabase.com](https://supabase.com) にアクセス → 「Start your project」
2. GitHubアカウントでサインアップ
3. 「New project」をクリック
4. 以下を設定：
   - **Name:** `pgvector-tutorial`
   - **Database Password:** 任意のパスワード（後で使うので控えておく）
   - **Region:** `Northeast Asia (Tokyo)`
5. 「Create new project」をクリック（2〜3分かかります）

### 1-2. pgvector拡張を有効化

Supabaseはデフォルトでpgvectorが使えますが、明示的に有効化します。

ダッシュボードの「SQL Editor」を開いて以下を実行：

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

### 1-4. 接続情報を取得

ダッシュボードの「Settings」→「Database」→「Connection parameters」から以下をコピー：

```
Host:     db.xxxxxxxxxxxx.supabase.co
Port:     5432
Database: postgres
User:     postgres
Password: （Step 1-1で設定したパスワード）
```

> **Supabaseの接続文字列（URI形式）も使えます:**
> `Settings` → `Database` → `Connection string` → `URI` からコピーできます

---

## Step 2: ローカルのデータをSupabaseに移行

ローカルのpgvectorに格納したドキュメントをSupabaseに移行します。

### 2-1. `.env` にSupabaseの接続情報を追加

`.env` ファイルを以下のように更新：

```
# Gemini API
GEMINI_API_KEY=AIza...

# ローカルDB（従来）
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password

# Supabase（新規追加）
SUPABASE_HOST=db.xxxxxxxxxxxx.supabase.co
SUPABASE_PORT=5432
SUPABASE_DB=postgres
SUPABASE_USER=postgres
SUPABASE_PASSWORD=（Supabaseのパスワード）
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

# ── Supabaseに接続 ────────────────────────────────────────────
supa_conn = psycopg2.connect(
    host=os.getenv("SUPABASE_HOST"),
    port=os.getenv("SUPABASE_PORT"),
    dbname=os.getenv("SUPABASE_DB"),
    user=os.getenv("SUPABASE_USER"),
    password=os.getenv("SUPABASE_PASSWORD"),
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

# 確認
supa_cur.execute("SELECT COUNT(*) FROM documents;")
count = supa_cur.fetchone()[0]
print(f"Supabase内のドキュメント数: {count}")

local_conn.close()
supa_conn.close()
```

```bash
python migrate_to_supabase.py
```

---

## Step 3: MCPサーバーをRender用に修正

### 3-1. Render用のサーバーファイルを作成 — `mcp_server/server_render.py`

`server_http.py` をベースに、DB接続をSupabaseに変更します。

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

# ── DB接続（環境変数から取得） ──────────────────────────────────
# ローカル: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# Render:  DATABASE_URL（Renderが自動設定、またはSupabaseのURLを設定）
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render / Supabase の接続文字列を使う
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
else:
    # ローカル用フォールバック
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST", os.getenv("DB_HOST")),
        port=os.getenv("SUPABASE_PORT", os.getenv("DB_PORT")),
        dbname=os.getenv("SUPABASE_DB", os.getenv("DB_NAME")),
        user=os.getenv("SUPABASE_USER", os.getenv("DB_USER")),
        password=os.getenv("SUPABASE_PASSWORD", os.getenv("DB_PASSWORD")),
        sslmode="require",
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
    port = int(os.getenv("PORT", 8000))  # RenderはPORT環境変数を自動設定
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )
```

### 3-2. `requirements.txt` を更新

```bash
pip freeze > requirements.txt
```

`requirements.txt` に以下が含まれていることを確認：

```
fastmcp
psycopg2-binary
google-genai
python-dotenv
uvicorn
```

### 3-3. `render.yaml` を作成（オプション）

プロジェクトルートに `render.yaml` を作るとRenderの設定をコード管理できます：

```yaml
# render.yaml
services:
  - type: web
    name: pgvector-mcp-server
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python mcp_server/server_render.py
    envVars:
      - key: GEMINI_API_KEY
        sync: false  # Renderのダッシュボードで手動設定
      - key: DATABASE_URL
        sync: false  # Renderのダッシュボードで手動設定
```

---

## Step 4: GitHubにプッシュ

```bash
git add .
git commit -m "feat: add Render deployment files

- Add server_render.py with DATABASE_URL support
- Add render.yaml for infrastructure as code
- Add migrate_to_supabase.py for data migration"

git push origin main
```

---

## Step 5: Renderにデプロイ

### 5-1. Renderアカウント作成

1. [render.com](https://render.com) にアクセス
2. 「Get Started for Free」→ GitHubでサインアップ
3. クレジットカード不要

### 5-2. Webサービスを作成

1. ダッシュボードの「New」→「Web Service」
2. 「Connect a repository」→ GitHubのリポジトリを選択
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
| `DATABASE_URL` | SupabaseのConnection URI |

> **SupabaseのDATABASE_URLの取得方法:**
> Supabaseダッシュボード → `Settings` → `Database` → `Connection string` → `URI`
> `postgresql://postgres:パスワード@db.xxxx.supabase.co:5432/postgres`

5. 「Save Changes」でデプロイが自動開始されます

### 5-4. デプロイ確認

デプロイが完了すると以下のようなURLが発行されます：

```
https://pgvector-mcp-server.onrender.com
```

MCPエンドポイントは：

```
https://pgvector-mcp-server.onrender.com/mcp
```

ブラウザでアクセスして応答があればデプロイ成功です。

---

## Step 6: Claude Desktopから接続

`~/Library/Application Support/Claude/claude_desktop_config.json` を更新：

```json
{
  "mcpServers": {
    "pgvector-search-remote": {
      "url": "https://pgvector-mcp-server.onrender.com/mcp"
    }
  }
}
```

Claude Desktopを再起動して、チャットで検索ツールが使えるか確認します：

```
「機械学習の評価指標について教えてください」
```

---

## スリープ対策（オプション）

Renderの無料サービスは15分間アクセスがないとスリープします。スリープを防ぐには定期的にpingを送ります。

### UptimeRobot（無料）で5分ごとにping

1. [uptimerobot.com](https://uptimerobot.com) に登録（無料）
2. 「Add New Monitor」→ HTTP(s)を選択
3. URL: `https://pgvector-mcp-server.onrender.com/mcp`
4. Monitoring Interval: `5 minutes`

これでスリープを防ぎ、常時起動状態を維持できます。

---

## よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `SSL connection required` | Supabase接続でSSLなし | `sslmode="require"` を追加 |
| `could not connect to server` | DATABASE_URLが間違っている | SupabaseのConnection URIを再確認 |
| `ModuleNotFoundError` | requirements.txtに不足 | `pip freeze > requirements.txt` して再push |
| デプロイ後に接続が遅い | Renderのスリープ | UptimeRobotで定期ping |
| Supabaseがポーズ状態 | 7日間アクセスなし | Supabaseダッシュボードから「Resume」 |

---

---

## 参考

- [Render公式ドキュメント](https://render.com/docs)
- [Supabase pgvectorドキュメント](https://supabase.com/docs/guides/database/extensions/pgvector)
- [FastMCP HTTPトランスポート](https://gofastmcp.com/deployment/running-server)