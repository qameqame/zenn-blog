---
title: "MCPツールにおける「検索コントラクト」の明示とは"
emoji: "📋"
type: "tech"
topics: ["mcp", "rag", "llm", "python", "設計"]
published: true
---

## はじめに

先日、pgvectorの検索機能をMCPサーバーとして公開する記事を[Dev.to](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-mcp-exposing-pgvector-as-a-reusable-tool-server-2onc)に投稿したところ、「検索コントラクト」に関するフィードバックコメントをいただきました。

> pgvector as an MCP tool is a nice boundary because retrieval becomes a reusable capability instead of being buried inside one app. I would make the retrieval contract explicit: corpus, filters, scoring, freshness, and why the returned chunks are safe to use for the current task.
>
> — [Alex Shev (@alexshev), Dev.to](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-mcp-exposing-pgvector-as-a-reusable-tool-server-2onc#comment-3a7jf)

**「検索コントラクト（retrieval contract）を明示すべき」** という指摘です。

Dev.toではAIでコメントを書いているケースも少なくなく、このコメントもAIっぽいです（とか言って、AIでなかったらスミマセン）が、興味深いコメントだったため、本記事では「検索コントラクト」とは何か・なぜ重要か・どう実装するかを整理します。

---

## "being buried inside one app" とは

コメントの中に `being buried inside one app` という表現があります。

「1つのアプリの中に埋もれている・閉じ込められている」という意味で、MCPでサーバー化する前の状態を指しています。

```python
# MCPなし → 1つのPythonスクリプトに閉じ込められている
def search_documents(query: str) -> list[dict]:
    ...  # このプロジェクトだけで使える

# MCPサーバー化 → どこからでも使える
@mcp.tool
def search_documents(query: str) -> list[dict]:
    ...  # Claude Desktop・Gemini・どのLLMからも使える
```

MCPでサーバー化することで「検索機能がインフラになった」一方で「インフラとして使うなら、呼び出し側への約束（コントラクト）を明示すべき」というフィードバックです。

---

## 検索コントラクトとは

コントラクト（契約）とは「このツールを呼ぶと、どんな条件でどんな結果が返ってくるかを明示した約束」です。

現在の `search_documents` の戻り値はこうなっています：

```python
# 現在の実装
@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """全カテゴリのドキュメントからクエリに関連するものを検索する"""
    ...
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]
```

これを呼ぶLLMには以下のことが **わかりません**：

- どのドキュメント群を検索したか（コーパス）
- フィルターが適用されているか
- `similarity: 0.78` は高いのか低いのか（スコアリングの意味）
- データはいつ時点のものか（鮮度）
- 返されたドキュメントは信頼して良いか（安全性）

---

## 5つのコントラクト要素

フィードバックで挙げられた5つの要素を順番に見ていきます。

### 1. corpus（コーパス）

「何のドキュメントを検索したか」の明示です。

```python
# 明示前
{"title": "...", "body": "...", "similarity": 0.78}

# 明示後
{
    "corpus": "internal-tech-docs-v2",  # ← 何を検索したか
    "results": [...]
}
```

LLMが回答を生成するとき「このデータは社内のものか・公開情報か」を知っていると、回答の信頼性判断に使えます。マルチテナント構成（会社Aと会社Bのデータが混在）では特に重要です。

### 2. filters（フィルター）

「どんな絞り込み条件で検索したか」の明示です。

```python
# search_by_category("ML") を呼んだ場合
{
    "filters_applied": {"category": "ML"},  # ← フィルターが効いている
    "results": [...]
}
```

フィルターが適用されていることをLLMが知らないと「他のカテゴリには情報がない」という誤解につながります。`filters_applied: {}` なら全体検索したことが明確になります。

### 3. scoring（スコアリング）

「類似度スコアの意味・計算方法」の明示です。

```python
{
    "scoring": {
        "method": "cosine-similarity",
        "embedding_model": "gemini-embedding-001",
        "dimensions": 768,
        "threshold": 0.5  # ← このスコア以上のみ返している
    },
    "results": [{"similarity": 0.78, ...}]
}
```

`similarity: 0.78` が高いのか低いのかは、スコアリングの方法を知らないとLLMには判断できません。閾値を明示することで「閾値未満のドキュメントは除外済み」という情報も伝わります。

### 4. freshness（鮮度）

「データがいつ時点のものか」の明示です。

```python
{
    "index_last_updated": "2026-06-28T00:00:00+09:00",
    "results": [
        {
            "title": "...",
            "created_at": "2026-01-15",  # ← ドキュメント自体の作成日
            "similarity": 0.78
        }
    ]
}
```

古いドキュメントを返しているのに「最新情報によると」という回答を生成するリスクを防げます。Observabilityのトレーシングでも鮮度を記録すると、「なぜ古い情報を返したか」の調査が簡単になります。

### 5. safety（安全性）

「返されたチャンクをこのタスクに使って良い理由」の明示です。

```python
{
    "safety": {
        "source": "internal-verified",    # 検証済み社内ドキュメント
        "access_level": "internal-only",  # アクセスレベル
        "pii_screened": True,             # 個人情報スクリーニング済み
        "guardrail_passed": True          # セキュリティチェック通過済み
    },
    "results": [...]
}
```

セキュリティ章で実装したガードレールの結果をここに含めることで、LLMが「このデータは信頼できる」と判断できます。

---

## 実装：コントラクト付き search_documents

```python
# mcp_server/server.py （更新版）
from datetime import datetime

@mcp.tool
def search_documents(query: str, top_k: int = 3) -> dict:
    """
    全カテゴリのドキュメントからクエリに関連するものを検索する。

    Args:
        query: 検索クエリ
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        corpus: 検索対象のドキュメント集合
        filters_applied: 適用されたフィルター条件
        scoring: スコアリングの方法と閾値
        index_freshness: インデックスの最終更新日時
        safety: 安全性情報
        results: 検索結果のリスト
    """
    query_embedding = get_embedding(query)

    cur.execute("""
        SELECT title, body, category, created_at,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))

    rows = cur.fetchall()
    results = [
        {
            "title": r[0],
            "body": r[1],
            "category": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "similarity": round(r[4], 4),
        }
        for r in rows
    ]

    # コントラクトを明示して返す
    return {
        "corpus": "internal-tech-docs",
        "filters_applied": {},  # 全体検索の場合は空
        "scoring": {
            "method": "cosine-similarity",
            "embedding_model": "gemini-embedding-001",
            "dimensions": 768,
        },
        "index_freshness": datetime.now().isoformat(),
        "safety": {
            "source": "internal-verified",
            "pii_screened": True,
        },
        "results": results,
    }
```

---

## なぜEvalsと繋がるのか

いただいたフィードバックに対して「Evals much more tractable」という返答をしましたが、これも重要なポイントです。

コントラクトが明示されると、各要素を独立して評価できます：

```python
# evals/eval_rag.py への追加
def eval_contract_completeness(search_result: dict) -> dict:
    """検索コントラクトの完全性を評価する"""
    required_fields = ["corpus", "filters_applied", "scoring", "index_freshness", "safety", "results"]
    present = [f for f in required_fields if f in search_result]

    return {
        "completeness": len(present) / len(required_fields),
        "missing_fields": [f for f in required_fields if f not in search_result],
    }
```

コントラクトが欠けている場合、Evalsのスコアが下がる仕組みにできます。

---

## まとめ

今回は「MCPでツールをサーバー化する」だけでなく「そのツールの動作を明示的に約束する」ことが、本番で信頼できるシステムを作る上で重要だと改めて学びました。

| 要素 | 効果 |
|------|------|
| corpus | LLMが「何のデータか」を判断できる |
| filters | 「検索範囲の制限」を透明化できる |
| scoring | スコアの意味をLLMが解釈できる |
| freshness | 古いデータによる誤答を防げる |
| safety | ガードレール通過情報をLLMに渡せる |


---

## 参考

- [Dev.to 記事: Building a RAG System from Scratch — MCP](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-mcp-exposing-pgvector-as-a-reusable-tool-server-2onc)
- [ソースコード: github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)
