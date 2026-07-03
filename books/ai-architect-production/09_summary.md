---
title: "まとめ — AIアーキテクトとして次のステップへ"
---

## このガイドで実装したこと

[前作](https://zenn.dev/hkame/books/ai-architect-rag)でRAGからクラウドデプロイまでを実装し、本ガイドではその先の「本番運用」を体系的に実装しました。

```
evals/
  dataset.py          # 評価データセット
  eval_rag.py         # Context Recall・Relevancy・Faithfulness

observability/
  traced_rag.py       # @observe()でRAGパイプラインをトレース
  traced_agent.py     # Agentの各ステップをトレース（Langfuse v4）

security/
  input_validator.py  # プロンプトインジェクション検知
  output_validator.py # 個人情報マスク・漏洩検知
  guardrails.py       # レート制限・ログ統合
  secure_rag.py       # ガードレール付きRAG

llmops/
  prompt_registry.py  # プロンプトv1.0〜v1.2のバージョン管理
  ci_eval.py          # 品質ゲート（Overall 75%以上でデプロイ）
  cost_tracker.py     # APIコスト追跡

finetuning/
  prepare_dataset.py  # Alpacaフォーマットに変換
  train_lora.py       # LoRA（r=8）でFine-tuning（CPUで2分）
  inference.py        # ベースモデルとの比較

multiagent/
  search_worker.py    # 検索専門ワーカー
  quality_worker.py   # 品質チェック専門ワーカー
  orchestrator.py     # タスク分解・結果統合
  14_multiagent.py    # 実行スクリプト

governance/
  ai_registry.py      # AIシステム台帳
  risk_assessor.py    # リスク評価（スコア0.18 → LOW）
  audit_logger.py     # 監査ログ（Article 12対応）
  compliant_rag.py    # AI開示文付きRAG（Article 50対応）
```

---

## 各章で学んだ設計判断

### 第2章: Evals

ルールベース（Context Recall・Answer Relevancy）とLLM-as-a-Judge（Faithfulness）を組み合わせることで、速度とコストのバランスが取れた評価が実現できます。

### 第3章: Observability（Langfuse v4）

`@observe()` デコレーターを追加するだけでトレースが記録されます。v4の重要な変更点は `load_dotenv()` の後に `get_client()` を呼ぶ順序が必須という点です。

### 第4章: Security

多層防御（Defense in Depth）が原則です。入力検証 → システムプロンプト → 出力検証 → レート制限の4層で守ります。

### 第5章: MLOps / LLMOps

GitHubにpushするたびにEvalsが自動実行され、品質基準（Overall 75%以上）を満たした場合のみRenderへ自動デプロイします。

### 第6章: Fine-tuning（LoRA）

全パラメータ（27億個）の0.09%（260万個）だけを学習します。CPUで2分以内に完了し、データは8件でも学習の傾向が確認できます。実用的な品質向上には100件以上が必要です。

### 第7章: マルチエージェント

単一責任の原則が鍵です。検索ワーカー・品質チェックワーカー・オーケストレーターがそれぞれ1つのことだけに集中することで、各AgentのプロンプトがシンプルになりLLMの出力品質が上がります。

### 第8章: ガバナンス

今回のRAGシステムはEU AI Actの「限定リスク（チャットボット）」に分類されます。リスクスコアは0.18（LOW）で、AIであることの開示（Article 50）と監査ログ（Article 12）を実装することでコンプライアンスの基盤が整います。

---

## 2冊の本を通じた全体像

```
【本1: AI初学者のためのRAG実装ガイド】
基礎 → RAG → Tool Use → Agents → MCP → デプロイ
「動くシステムを作る」

【本2: AIシステムの本番運用ガイド（本書）】
Evals → Observability → Security → MLOps
→ Fine-tuning → マルチエージェント → ガバナンス
「本番で使えるシステムにする」
```

AIアーキテクトの仕事は「動くものを作る」だけでなく、「品質を測定し」「動作を可視化し」「攻撃から守り」「継続的に改善し」「複数のAgentを協調させ」「規制に対応する」仕組みを設計することです。

---

## 実装した全ファイルの一覧

| フェーズ | ファイル数 | 主要技術 |
|---------|-----------|---------|
| Evals | 2 | LLM-as-a-Judge |
| Observability | 2 | Langfuse v4 |
| Security | 4 | 正規表現・ガードレール |
| MLOps | 3 | GitHub Actions・プロンプト管理 |
| Fine-tuning | 3 | LoRA・Hugging Face |
| マルチエージェント | 4 | オーケストレーター・ワーカー |
| ガバナンス | 4 | EU AI Act・監査ログ |
| **合計** | **22** | |

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

---

## 参考リンク

- [前作: AI初学者のためのRAG実装ガイド](https://zenn.dev/hkame/books/ai-architect-rag)
- [Langfuse公式ドキュメント（v4）](https://langfuse.com/docs)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [EU AI Act 公式テキスト](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [FastMCP ドキュメント](https://gofastmcp.com)
