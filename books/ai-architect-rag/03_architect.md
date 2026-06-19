---
title: "AIアーキテクト視点で読み解くRAG設計"
---

## はじめに

[第2章](./02_rag)では、pgvectorとGeminiを使ってRAGパイプラインを実装しました。本記事ではこの記事では、ハンズオンで実装した各コンポーネントをアーキテクト視点で読み直し、設計判断の背景と本番運用への拡張ポイントを整理します。

---

## 全体アーキテクチャの再整理

まずハンズオンで作ったシステムの全体像を、アーキテクト視点で整理します。

```
┌─────────────────────────────────────────────────────┐
│                   RAGパイプライン                      │
│                                                     │
│  ① 格納フェーズ（03_ingest.py）                        │
│  テキスト → [Gemini Embedding] → ベクトル → pgvector  │
│                                                     │
│  ② 検索フェーズ（04_search.py）                        │
│  クエリ → [Gemini Embedding] → ベクトル検索 → 上位K件  │
│                                                     │
│  ③ 生成フェーズ（05_rag.py）                           │
│  検索結果 + クエリ → [Gemini LLM] → 回答              │
└─────────────────────────────────────────────────────┘
```

システムには **3つの役割** が存在します。

| コンポーネント | 役割 | 今回の実装 |
|--------------|------|-----------|
| Embeddingモデル | テキスト → ベクトルの変換 | gemini-embedding-001 |
| Vector DB | ベクトルの保存・近傍探索 | pgvector（PostgreSQL拡張） |
| LLM | 検索結果を自然言語で回答 | gemini-2.5-flash |

この3つが独立して動いていることが、RAGアーキテクチャの重要な特徴です。それぞれを別のコンポーネントとして分離しているため、将来的にどれか一つを差し替えることができます。

---

## 設計判断① なぜGeminiをEmbeddingとLLMの両方に使うのか

ハンズオンではGeminiを2つの用途で使いました。

```python
# Embeddingモデル（意味のベクトル化）
client.models.embed_content(model="gemini-embedding-001", ...)

# LLM（回答生成）
client.models.generate_content(model="gemini-2.5-flash", ...)
```

**設計上の理由は「シンプルさと無料枠の活用」**です。

ただしアーキテクト視点では、EmbeddingモデルとLLMは **別々のコンポーネントとして独立させる** のが正しい設計です。理由は以下の通りです。

```
Embeddingモデルに求められる性質:
  → 高精度なベクトル変換
  → 同じテキストは常に同じベクトルを返す（決定論的）
  → 一度格納したら変えにくい（変えると再インデックスが必要）

LLMに求められる性質:
  → 自然な文章生成
  → コスト・レイテンシのバランス
  → 用途に応じて差し替えやすい
```

本番システムでは、Embeddingモデルは固定してVector DBへの格納と検索で一貫性を保ち、LLMは要件に応じてモデルを変更する設計が一般的です。

---

## 設計判断② なぜpgvectorを選んだのか

ハンズオンではPineconeやWeaviateではなくpgvectorを選びました。この選択に込められた設計判断を整理します。

```
Vector DBの選択肢:

専用Vector DB（Pinecone, Qdrant, Weaviate）
  ✓ 大規模に対応（億単位のベクトル）
  ✓ 自動スケーリング
  ✗ 既存DBと別管理
  ✗ コスト発生

pgvector（PostgreSQL拡張）
  ✓ 既存PostgreSQLに追加するだけ
  ✓ SQLで結合・フィルタが自由
  ✓ 無料（ローカル）
  ✗ 数百万件を超えると専用DBに劣る
```

**pgvectorが適切な判断となる条件：**

- ドキュメント数が数百万件以下
- 既存システムにPostgreSQLが使われている
- 通常のDBクエリとベクトル検索を組み合わせたい

本番で億単位のドキュメントを扱う場合は、Pinecone・Qdrantへの移行を検討します。ただし今回のようなプロトタイプや中規模システムではpgvectorは十分な選択肢です。

---

## 設計判断③ なぜ次元数を768に削減したのか

ハンズオンの中で試行錯誤したところがこの箇所です。

```python
config=types.EmbedContentConfig(
    output_dimensionality=768,  # 本来は3072次元を768に削減
)
```

```sql
-- HNSWインデックスは2000次元までの制限がある
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
```

**アーキテクト視点での判断：**

| 次元数 | 精度 | ストレージ | 検索速度 | HNSWサポート |
|--------|------|-----------|---------|------------|
| 3072（フル） | 最高 | 大 | 遅い | ✗（pgvector制限） |
| 768（削減） | 高い | 中 | 速い | ✓ |
| 256（削減） | やや低下 | 小 | 最速 | ✓ |

実務では **768次元で精度と速度のバランスが取れる** ことが多く、多くの本番システムで採用される次元数です。次元削減はEmbeddingの品質を落とすのではなく、重要な情報を圧縮して保持します。

---

## 設計判断④ task_typeの使い分けがなぜ重要なのか

```python
# 格納時
task_type="RETRIEVAL_DOCUMENT"

# 検索時
task_type="RETRIEVAL_QUERY"
```

これはGemini固有の機能ですが、アーキテクト視点では **「非対称な類似度計算」** の概念として理解します。

```
問題: ドキュメントとクエリは性質が異なる
  ドキュメント: 長い・詳細・情報が凝縮されている
  クエリ:       短い・曖昧・意図を表現している

解決: 用途に応じてEmbeddingの最適化方向を変える
  RETRIEVAL_DOCUMENT → 「情報を提供する側」として最適化
  RETRIEVAL_QUERY    → 「情報を探す側」として最適化
```

これを揃えないと検索精度が落ちます。格納時と検索時で異なる `task_type` を使うことは、RAGシステムの精度を上げるための基本的な設計パターンです。

---

## 設計判断⑤ メタデータフィルタの設計思想

ハンズオンでは `category` によるフィルタを実装しました。

```python
where_clause += " AND category = %s"
params = [query_embedding, category, query_embedding]
```

**なぜフィルタが必要か：**

ベクトル検索は「意味の近さ」で探すため、カテゴリが異なっていても似た意味の文書がヒットすることがあります。

```
クエリ: 「Pythonでモデルを評価したい」

フィルタなし: Cloud系・ML系・Python系が混在して返ってくる可能性
フィルタあり: MLカテゴリのみに絞ってから意味検索 → 精度向上
```

**本番設計では以下のメタデータを設計する：**

```sql
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    title       TEXT,
    body        TEXT,
    category    TEXT,        -- 大分類フィルタ
    department  TEXT,        -- 部門フィルタ（マルチテナント）
    language    TEXT,        -- 言語フィルタ
    created_at  TIMESTAMP,   -- 日付フィルタ（最新情報優先）
    version     INTEGER,     -- バージョン管理
    embedding   vector(768)
);

-- メタデータ列にB-treeインデックスを追加（フィルタ高速化）
CREATE INDEX ON documents (category);
CREATE INDEX ON documents (department);
CREATE INDEX ON documents (created_at);
```

メタデータ設計はRAGの精度を左右する重要な設計判断です。事前に「どんな条件で絞り込みたいか」を洗い出してスキーマに反映することがアーキテクトの仕事です。

---

## 本番運用で変わる3つのポイント

ハンズオンの実装はプロトタイプとして十分ですが、本番では以下の3点が変わります。

### 1. バッチEmbedding生成

```python
# ハンズオン: 1件ずつ処理
for doc in sample_docs:
    embedding = get_embedding(doc["text"])  # APIを毎回呼ぶ

# 本番: バッチ処理でAPIコストと時間を削減
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[doc["text"] for doc in batch],  # まとめて送る
)
```

### 2. 接続プール管理

```python
# ハンズオン: スクリプト実行のたびに接続
conn = psycopg2.connect(...)

# 本番: 接続プールで効率化
from psycopg2 import pool
connection_pool = pool.SimpleConnectionPool(1, 20, **db_config)
```

### 3. Embeddingのキャッシュ

```python
# 同じテキストを毎回Embeddingすると無駄なAPIコストが発生する
# 本番ではRedisなどでキャッシュする

import hashlib
import json

def get_embedding_with_cache(text: str, cache: dict) -> list[float]:
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]
    embedding = get_embedding(text)
    cache[cache_key] = embedding
    return embedding
```

---

## アーキテクチャの進化パス

今回実装したシステムは「基本的なRAG」です。アーキテクト視点では、要件に応じて以下のように進化させます。

```
レベル1（今回実装）: 基本RAG
  テキスト検索 → LLM回答

レベル2: 精度向上
  ハイブリッド検索（ベクトル + BM25キーワード）
  再ランキング（Cross-Encoder）
  チャンク分割の最適化

レベル3: 自律化
  Tool Use（LLMが自律的に検索）
  AI Agents（複数ツールの組み合わせ）
  ← 次回の記事のテーマ

レベル4: 本番化
  評価（Evals）
  Observability（トレーシング・コスト管理）
  マルチテナント対応
```

---

## まとめ

ハンズオンで実装した各コンポーネントをアーキテクト視点で振り返ると、以下の設計判断が含まれていました。

| 設計判断 | 選択 | 理由 |
|---------|------|------|
| Vector DB | pgvector | 既存PostgreSQLと統合・中規模に十分 |
| 次元数 | 768 | HNSWの制限回避・精度と速度のバランス |
| task_type | 格納/検索で分離 | 非対称な類似度計算で精度向上 |
| メタデータ | categoryフィルタ | ベクトル検索の精度を絞り込みで補完 |
| LLM | gemini-2.5-flash | 無料枠・回答生成に十分な性能 |

次回は今回のRAGシステムに **Tool Use** を組み込み、LLMが自律的に検索するエージェント設計を解説します。

---

## 参考

- [前回記事: pgvectorとGeminiで作るRAGパイプライン](https://zenn.dev/qame/articles/ad2fafe66a50ad)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Gemini Embedding API ドキュメント](https://ai.google.dev/gemini-api/docs/embeddings)