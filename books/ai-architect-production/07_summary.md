---
title: "まとめ — AIアーキテクトとして次のステップへ"
---

## このガイドで実装したこと

[前作](https://zenn.dev/qame/books/ai-architect-rag)でRAGからクラウドデプロイまでを実装し、本ガイドではその先の「本番運用」を体系的に実装しました。

```
evals/
  dataset.py          # 評価データセット（5〜7件）
  eval_rag.py         # Context Recall・Relevancy・Faithfulness

observability/
  traced_rag.py       # @observe()でRAGパイプラインをトレース
  traced_agent.py     # Agentの各ステップをトレース

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
  train_lora.py       # LoRA（r=8）でFine-tuning
  inference.py        # ベースモデルとの比較
```

---

## 各章で学んだ設計判断

### 第2章: Evals

| 指標 | 手法 | 測ること |
|------|------|---------|
| Context Recall | ルールベース | 正しいドキュメントを検索できているか |
| Answer Relevancy | ルールベース | 期待するキーワードが回答に含まれているか |
| Faithfulness | LLM-as-a-Judge | ハルシネーションがないか |

ルールベースとLLM-as-a-Judgeを組み合わせることで、速度とコストのバランスが取れた評価が実現できます。

### 第3章: Observability

`@observe()` デコレーターを既存のコードに追加するだけでトレースが記録されます。Langfuse v4の重要な変更点として `load_dotenv()` の後に `get_client()` を呼ぶ順序が必須です。

### 第4章: Security

多層防御（Defense in Depth）が原則です。

```
Layer 1: 入力検証          ← 悪意ある入力をシステムに入れない
Layer 2: システムプロンプト ← LLMに制約を設ける
Layer 3: 出力検証          ← 問題ある出力をユーザーに見せない
Layer 4: レート制限        ← 大量攻撃を防ぐ
```

### 第5章: MLOps / LLMOps

GitHubにpushするたびに自動で評価が走るパイプラインが完成しました。品質基準（Overall 75%以上）を満たした場合のみRenderへ自動デプロイします。

### 第6章: Fine-tuning

LoRAの核心は「全パラメータ（27億個）の0.09%（260万個）だけを学習する」ことです。CPUで2分以内に完了し、モデルの基本的な振る舞いを変えずに特定のタスクに特化できます。

---

## 2冊の本を通じた全体像

```
【本1: AI初学者のためのRAG実装ガイド】
基礎 → RAG → Tool Use → Agents → MCP → デプロイ
「動くシステムを作る」

【本2: AIシステムの本番運用ガイド（本書）】
Evals → Observability → Security → MLOps → Fine-tuning
「本番で使えるシステムにする」
```

AIアーキテクトの仕事は「動くものを作る」だけでなく、「品質を測定し」「動作を可視化し」「攻撃から守り」「継続的に改善する」仕組みを設計することです。

---

## 次に学ぶべきこと

本ガイドを完了した後の次のステップは3つあります。

**マルチエージェント設計**
複数のAgentが協調してタスクを分担するアーキテクチャです。オーケストレーターAgentがサブAgentを管理し、並列処理や専門化が可能になります。

**ガバナンス / 倫理**
EU AI Actへの対応・リスク管理・監査ログ・バイアス検出など、組織としてAIを運用するための仕組みです。

**アーキテクチャパターン**
ハイブリッドRAG（ベクトル検索 + キーワード検索）・RAG + Fine-tuningの組み合わせ・コスト最適化パターンなど、実務で使われる設計パターンの体系的な理解です。

---

## 参考リンク

- [前作: AI初学者のためのRAG実装ガイド](https://zenn.dev/qame/books/ai-architect-rag)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Langfuse公式ドキュメント（v4）](https://langfuse.com/docs)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [FastMCP ドキュメント](https://gofastmcp.com)
