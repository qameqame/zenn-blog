---
title: "はじめに — 本番運用とは何か"
---

## このガイドについて

[前作「AI初学者のためのRAG実装ガイド」](https://zenn.dev/qame/books/ai-architect-rag)では、pgvectorとGeminiを使ってRAGシステムをゼロから実装し、Tool Use・AI Agents・MCP・クラウドデプロイまで一気通貫で進めました。

本ガイドはその続編です。「動くシステムを作る」から「本番で使えるシステムにする」へ、次のステップを扱います。

```
【前作で実装したこと】
RAG → Tool Use → AI Agents → MCP → Render × Supabase デプロイ

【本ガイドで学ぶこと】
Evals → Observability → Security → MLOps → Fine-tuning
```

---

## なぜ本番運用が難しいのか

RAGやAgentを実装したあと、本番で使おうとすると必ず以下の問題に直面します。

**品質の問題**
「回答が正しいかどうか」を手動で確認し続けることはスケールしません。自動で品質を測定する仕組みが必要です。

**可視性の問題**
本番で何か問題が起きたとき「どのステップで何が起きたか」が追跡できなければ原因究明ができません。

**セキュリティの問題**
外部ユーザーからのリクエストを受け付けるとき、プロンプトインジェクションや禁止コンテンツへの対処が必要です。

**継続的改善の問題**
プロンプトを改善するたびに「本当に良くなったか」を確認する仕組みがなければ、改善が進みません。

**モデルの問題**
汎用モデルではなく、特定のドメインに特化したモデルを使いたい場合があります。

---

## 本ガイドの構成

各章は独立して読めます。前作の実装（pgvector・Gemini・RAG）を前提としていますが、概念の理解だけであれば単体でも読めます。

| 章 | テーマ | 解決する問題 |
|----|--------|------------|
| 第2章 | Evals | 回答品質の自動測定 |
| 第3章 | Observability | トレーシング・コスト管理 |
| 第4章 | Security | ガードレール・攻撃対策 |
| 第5章 | MLOps / LLMOps | CI/CD・プロンプト管理 |
| 第6章 | Fine-tuning | モデルのドメイン特化 |

---

## 前提条件

- [前作](https://zenn.dev/qame/books/ai-architect-rag)のpgvectorチュートリアル完了済み
- Python 3.11・Docker・pgvector の環境が構築済み
- `.env` に `GEMINI_API_KEY` が設定済み

---

## 使うツール

| ツール | 用途 | 無料枠 |
|--------|------|--------|
| Google Gemini API | LLM・Embedding | 1,500リクエスト/日 |
| pgvector | ベクトルDB | 無制限（ローカル） |
| Langfuse | Observabilityツール | クラウド版無料あり |
| GitHub Actions | CI/CDパイプライン | 2,000分/月（無料） |
| Hugging Face | Fine-tuningモデル | 無料 |

では第2章のEvalsから始めましょう。
