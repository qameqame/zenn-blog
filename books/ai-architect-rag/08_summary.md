---
title: "まとめ — このガイドで実装したこと"
---

## このガイドで実装したこと

本ガイドでは、ローカルのRAG実装からクラウドデプロイまで、一気通貫で実装しました。

```
01_setup_db.py       → pgvectorのテーブル作成
02_create_index.py   → HNSWインデックス構築
03_ingest.py         → Embeddingを生成してDBに格納
04_search.py         → ベクトル検索・フィルタ検索
05_rag.py            → RAGパイプライン完成
06_tool_basic.py     → Tool Use基本（LLMがツールを自律判断）
07_tool_multi.py     → 複数ツールの使い分け
08_tool_agent.py     → Agenticループ（複数ステップの自律実行）
09_agent_basic.py    → 複数ツールを持つエージェント
10_agent_memory.py   → メモリ機能付きエージェント
11_agent_planner.py  → Plan→Execute→Evaluate
mcp_server/server.py → MCPサーバー（stdioモード）
server_http.py       → MCPサーバー（HTTPモード）
server_render.py     → Renderデプロイ用サーバー
```

---

## 各章で学んだ設計判断

### 第2章: RAG実装

| 設計判断 | 選択 | 理由 |
|---------|------|------|
| Vector DB | pgvector | 既存PostgreSQLと統合・中規模に十分 |
| 次元数 | 768 | HNSWの制限回避・精度と速度のバランス |
| task_type | 格納/検索で分離 | 非対称な類似度計算で精度向上 |
| LLM | gemini-2.5-flash | 無料枠・回答生成に十分な性能 |

### 第3章: 各コンポーネントの役割

| コンポーネント | 役割 | 本番での注意点 |
|--------------|------|--------------|
| Embeddingモデル | テキスト→ベクトル変換 | 一度決めたら変えにくい（再インデックスが必要） |
| Vector DB | ベクトルの保存・検索 | スケールに応じてpgvector→専用DBへ移行 |
| LLM | 回答生成 | 要件に応じて差し替えやすい |

### 第4章: Tool Use

```
06: ツールを使うか使わないかをLLMが判断
07: 複数ツールからどれを使うかをLLMが判断（descriptionが鍵）
08: 何回ツールを使うか・いつ終わるかをLLMが判断（Agenticループ）
```

### 第5章: AI Agents

```
09: ReActパターン（Reasoning → Acting → Observation）
10: Long-term Memory（ファイルへの永続化）
11: Plan → Execute → Evaluate の3フェーズ設計
```

### 第6章: MCP

| 項目 | Tool Use | MCP |
|------|----------|-----|
| ツール定義 | FunctionDeclarationを手書き | @mcp.toolで自動生成 |
| ツール実行 | dispatch()で直接呼ぶ | call_tool()経由 |
| 再利用性 | プロジェクト内のみ | どのLLMからも使える |

### 第7章: クラウドデプロイ

```
MCPサーバー → Render（無料・永続・スリープあり）
pgvector DB → Supabase（無料・永続・Connection Pooler経由）
接続方式   → IPv6問題を回避するためポート6543を使用
```

---

## AI技術者としての学習ロードマップ

本ガイドで「応用」フェーズの3つを実装しました。次のステップを示します。

```
✅ 基礎フェーズ
   LLMの仕組み / Prompt Engineering / API・SDK

✅ 応用フェーズ（本ガイドで実装）
   RAG / Embedding / Vector DB

✅ 応用続き（本ガイドで実装）
   Fine-tuning / AI Agents / Tool Use

✅ 設計フェーズ（本ガイドで実装）
   AIシステム設計 / MCP / クラウドデプロイ

→ 次に学ぶこと
   評価（Evals）         ← Agentの回答品質を自動測定
   Observability         ← トレーシング・コスト管理
   セキュリティ          ← ガードレール・プロンプト注入対策
   MLOps / LLMOps       ← CI/CD・モデル管理・デプロイ自動化
   ガバナンス / 倫理      ← AI戦略・リスク管理・規制対応
```

---

## 本ガイドで構築した最終的なシステム構成

```
【ローカル開発環境】
Claude Desktop
    ↓ stdio
MCPサーバー（mcp_server/server.py）
    ↓ psycopg2
pgvector DB（Docker）
    ↓
Gemini Embedding / LLM

【クラウド環境】
13_mcp_http_agent.py
    ↓ HTTPS
Render（server_render.py）
    ↓ PostgreSQL（Connection Pooler / port 6543）
Supabase（pgvector）
    ↓
Gemini Embedding / LLM
```

---

## 後続編

本書の続編として、RAGシステムを「本番で使えるシステム」にするための技術を体系的に解説しています。

**[AIアーキテクトのための本番運用ガイド](https://zenn.dev/hkame/books/ai-architect-production)**

| 章 | 内容 |
|----|------|
| 第2章 | Evals — RAGの回答品質を自動測定する |
| 第3章 | Observability — Langfuse v4でトレース |
| 第4章 | Security — ガードレール・プロンプトインジェクション対策 |
| 第5章 | MLOps / LLMOps — CI/CDパイプライン |
| 第6章 | Fine-tuning — LoRAでモデルを特化させる |
| 第7章 | マルチエージェント — オーケストレーター設計 |
| 第8章 | ガバナンス — EU AI Act対応 |

## 参考リンク

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Google Gemini Embedding API](https://ai.google.dev/gemini-api/docs/embeddings)
- [FastMCP ドキュメント](https://gofastmcp.com)
- [MCP 公式仕様](https://modelcontextprotocol.io)
- [Render ドキュメント](https://render.com/docs)
- [Supabase pgvector ガイド](https://supabase.com/docs/guides/database/extensions/pgvector)