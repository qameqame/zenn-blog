---
title: "RAG・Embedding・Vector DBの実装"
---

## 環境セットアップ

### 前提条件

- pyenv導入済み
- Docker導入済み
- Google アカウント（Gemini APIキー取得用）

### 利用するツール

今回はEmbeddingの生成・回答生成に用いるLLMはGoogle Gemini APIの無料枠を利用し、Vector DBはDockerコンテナで稼働させる構成で進めます。

| ツール | 用途 | 無料枠 |
|--------|------|--------|
| Google Gemini API | Embeddingの生成・回答生成 | 1,500リクエスト/日 |
| pgvector（PostgreSQL拡張） | ベクトルの保存・検索 | 無制限（ローカル） |
| Docker | pgvectorの起動 | 無制限 |
| Python 3.11 | 実装言語 | — |

### 1-1. プロジェクトのPythonバージョンを固定

```bash
mkdir pgvector-tutorial && cd pgvector-tutorial

pyenv local 3.11.9
python --version
# => Python 3.11.9
```

> `3.11.9` がインストールされていない場合は先に `pyenv install 3.11.9` を実行してください。

### 1-2. 仮想環境の作成と有効化

```bash
python -m venv .venv
source .venv/bin/activate
# プロンプトが (.venv) になれば成功
```

### 1-3. ライブラリのインストール

```bash
pip install --upgrade pip
pip install psycopg2-binary google-genai python-dotenv
pip freeze > requirements.txt
```

> **注意:** `google-generativeai`（旧パッケージ）ではなく `google-genai`（新パッケージ）を使います。

### 1-4. DockerでpgvectorつきPostgreSQLを起動

```bash
docker run -d \
  --name pgvector-demo \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

起動確認：

```bash
docker exec -it pgvector-demo psql -U postgres -d vectordb -c "SELECT version();"
```

### 1-5. 環境変数の設定（`.env`ファイル）

```
GEMINI_API_KEY=AIza...
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password
```

**Gemini APIキーの取得手順:**

[aistudio.google.com](https://aistudio.google.com) にアクセスして以下の手順で取得できます。

1. Googleアカウントでログイン
2. 左サイドバーの「Get API key」をクリック
3. 「Create API key」をクリック
4. 表示された `AIza...` から始まるキーをコピー

---

## 実装

### ディレクトリ構成

上から順番に実行していきます。

```
pgvector-tutorial/
├── .env
├── .python-version      # pyenv local で自動生成
├── requirements.txt
├── .venv/
│
├── 01_setup_db.py       # テーブル作成
├── 02_create_index.py   # HNSWインデックス作成
├── 03_ingest.py         # ドキュメント格納
├── 04_search.py         # 基本検索 + フィルタ検索
└── 05_rag.py            # RAGパイプライン
```

### 1. データベースの準備 — `01_setup_db.py`

```python
# 01_setup_db.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# pgvector拡張を有効化（初回のみ必要）
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# ドキュメントテーブルを作成
# vector(768) = gemini-embedding-001 を output_dimensionality=768 で削減した次元数
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id          SERIAL PRIMARY KEY,
        title       TEXT NOT NULL,
        body        TEXT NOT NULL,
        category    TEXT,
        created_at  TIMESTAMP DEFAULT NOW(),
        embedding   vector(768)
    );
""")

conn.commit()
cur.close()
conn.close()

print("テーブル作成完了")
```

```bash
python 01_setup_db.py
# => テーブル作成完了
```

確認：

```bash
docker exec -it pgvector-demo psql -U postgres -d vectordb -c "\dx"
# => vector 0.8.x が表示されればOK
```

> **次元数について:** `gemini-embedding-001` の出力は本来3072次元ですが、pgvectorのHNSWインデックスは2000次元までの制限があるため、`output_dimensionality=768` で削減して使います。

### 2. HNSWインデックスの作成 — `02_create_index.py`

```python
# 02_create_index.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

cur.execute("""
    CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (
        m = 16,
        ef_construction = 64
    );
""")

conn.commit()
print("インデックス作成完了")
```

```bash
python 02_create_index.py
# => インデックス作成完了
```

> **補足:** `m` と `ef_construction` の目安
>
> | 用途 | m | ef_construction |
> |------|---|-----------------|
> | 開発・テスト | 8 | 32 |
> | 本番（標準） | 16 | 64 |
> | 高精度要求 | 32 | 128 |

### 3. ドキュメントの格納 — `03_ingest.py`

```python
# 03_ingest.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

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

def get_embedding(text: str) -> list[float]:
    """テキストをEmbeddingベクトルに変換する"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def insert_document(title: str, body: str, category: str) -> int:
    """ドキュメントをEmbeddingと一緒に格納する"""
    text_to_embed = f"{title}\n\n{body}"
    embedding = get_embedding(text_to_embed)

    cur.execute("""
        INSERT INTO documents (title, body, category, embedding)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """, (title, body, category, embedding))

    doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id

sample_docs = [
    {
        "title": "機械学習モデルの評価指標",
        "body": "精度、再現率、F1スコアなどの評価指標について解説します。"
                "分類問題では混同行列を使って各指標を計算します。",
        "category": "ML",
    },
    {
        "title": "scikit-learnによるモデル評価",
        "body": "Pythonのscikit-learnライブラリを使ってモデルを評価する方法。"
                "cross_val_scoreやclassification_reportの使い方を説明します。",
        "category": "ML",
    },
    {
        "title": "Pandasによるデータ前処理",
        "body": "欠損値処理、型変換、外れ値の扱い方を説明します。"
                "DataFrameの基本操作とクリーニング手順を解説します。",
        "category": "Python",
    },
    {
        "title": "AWSコスト最適化の実践",
        "body": "EC2インスタンスタイプの選定、スポットインスタンス活用、"
                "不要リソースの削除によるコスト削減手法を紹介します。",
        "category": "Cloud",
    },
    {
        "title": "Kubernetes Podの基本",
        "body": "Kubernetesにおける最小デプロイ単位であるPodの概念と、"
                "YAMLによるマニフェスト定義の書き方を解説します。",
        "category": "Cloud",
    },
]

for doc in sample_docs:
    doc_id = insert_document(doc["title"], doc["body"], doc["category"])
    print(f"格納完了: id={doc_id} / {doc['title']}")
```

```bash
python 03_ingest.py
# => 格納完了: id=1 / 機械学習モデルの評価指標
# => 格納完了: id=2 / scikit-learnによるモデル評価
# => 格納完了: id=3 / Pandasによるデータ前処理
# => 格納完了: id=4 / AWSコスト最適化の実践
# => 格納完了: id=5 / Kubernetes Podの基本
```

> **`task_type` について:** Gemini Embeddingは格納側とクエリ側で指定を分けることで精度が上がります。
>
> | task_type | 用途 |
> |-----------|------|
> | `RETRIEVAL_DOCUMENT` | DBに格納するドキュメント側 |
> | `RETRIEVAL_QUERY` | 検索クエリ側 |

### 4. ベクトル検索（基本 + フィルタ） — `04_search.py`

```python
# 04_search.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

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
        SELECT id, title, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]

def search_with_filter(query: str, category: str = None, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    where_clause = "WHERE 1=1"
    params = [query_embedding, query_embedding]
    if category:
        where_clause += " AND category = %s"
        params = [query_embedding, category, query_embedding]
    params.append(top_k)
    cur.execute(f"""
        SELECT id, title, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, params)
    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]

# 基本検索
results = search("機械学習のモデル精度を測る方法", top_k=3)
for r in results:
    print(f"[{r['similarity']:.4f}] {r['title']} ({r['category']})")

print("---")

# フィルタ付き検索（MLカテゴリのみ）
results = search_with_filter("Pythonでモデルを評価したい", category="ML", top_k=3)
for r in results:
    print(f"[{r['similarity']:.4f}] {r['title']} ({r['category']})")
```

```bash
python 04_search.py
# => [0.7806] 機械学習モデルの評価指標 (ML)
# => [0.7423] scikit-learnによるモデル評価 (ML)
# => [0.6015] Pandasによるデータ前処理 (Python)
# => ---
# => [0.7652] 機械学習モデルの評価指標 (ML)
# => [0.7341] scikit-learnによるモデル評価 (ML)
```

### 5. RAGパイプライン — `05_rag.py`

EmbeddingもLLMもGeminiで統一することで、完全無料でRAGを動かせます。

```python
# 05_rag.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

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

def search_with_filter(query: str, category: str = None, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    where_clause = "WHERE 1=1"
    params = [query_embedding, query_embedding]
    if category:
        where_clause += " AND category = %s"
        params = [query_embedding, category, query_embedding]
    params.append(top_k)
    cur.execute(f"""
        SELECT id, title, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, params)
    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]

def get_body(doc_id: int) -> str:
    cur.execute("SELECT body FROM documents WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    return row[0] if row else ""

def rag_answer(question: str, category: str = None) -> str:
    docs = search_with_filter(question, category=category, top_k=3)

    if not docs:
        return "関連するドキュメントが見つかりませんでした。"

    context = "\n\n".join([
        f"【{d['title']}】\n{get_body(d['id'])}"
        for d in docs
    ])

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

# 実行
answer = rag_answer("F1スコアはどう計算しますか？")
print(answer)
```

```bash
python 05_rag.py
```

> **無料枠の目安（2026年6月時点）:**
>
> | モデル | 無料枠 |
> |--------|--------|
> | `gemini-embedding-001` | 1,500 リクエスト/日、15 RPM |
> | `gemini-2.5-flash` | モデルにより異なる（AI Studioで確認） |

---

## 動作確認チェックリスト

```bash
# 1. pgvectorが有効になっているか確認
docker exec -it pgvector-demo psql -U postgres -d vectordb -c "\dx"

# 2. テーブルの中身を確認
docker exec -it pgvector-demo psql -U postgres -d vectordb \
  -c "SELECT id, title, category FROM documents;"

# 3. インデックスが作成されているか確認
docker exec -it pgvector-demo psql -U postgres -d vectordb \
  -c "\d documents"

# 4. embeddingの次元数を確認
docker exec -it pgvector-demo psql -U postgres -d vectordb \
  -c "SELECT id, title, vector_dims(embedding) AS dims FROM documents LIMIT 3;"
# => dims = 768 になっていればOK
```