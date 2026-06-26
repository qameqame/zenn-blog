---
title: "MLOps / LLMOps — CI/CDパイプラインで品質を継続的に守る"
---

## はじめに

[第4章のSecurity](./04_security)まで、Evals・Observability・Securityを個別に実装してきました。本章ではこれらを「継続的に運用する仕組み」として統合します。

LLMOpsはMLOpsとDNAを共有しますが、根本的に異なる問題を抱えています。プロンプトはコード、Evalsはユニットテストの代替、プロバイダー切り替えは日常的、コストは予測不能です。

```
【今まで】手動で実行するスクリプト群
python evals/eval_rag.py  ← 手動実行
python security/secure_rag.py  ← 手動実行

【今回 LLMOps】
GitHubにpushするたびに自動実行
  → Evalsで品質チェック
  → セキュリティ検証
  → 品質基準を満たしたら本番にデプロイ
```

2026年のMLOps成熟度モデルでは、Level 2（CI/CDパイプライン自動化）が最も高いROIをもたらすとされており、ほとんどの組織はLevel 1〜2の間にいます。

---

## LLMOpsがMLOpsと異なる点

LLMOpsはMLOps固有の懸念事項を追加します：プロンプトのバージョン管理と評価、ハルシネーション監視、RAG検索品質の測定、トークンコスト管理、コンテンツ安全監視です。

| MLOps | LLMOps |
|-------|--------|
| モデルファイルをバージョン管理 | プロンプトをバージョン管理 |
| 精度・損失でテスト | Evals（LLM-as-a-Judge）でテスト |
| モデルをデプロイ | プロンプト設定をデプロイ |
| データドリフトを監視 | 回答品質・コストを監視 |

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル（01〜13、evals/、observability/、security/）
│
├── llmops/
│   ├── prompt_registry.py    # ★ プロンプトのバージョン管理
│   ├── ci_eval.py            # ★ CI用の評価スクリプト
│   └── cost_tracker.py       # ★ APIコスト追跡
│
└── .github/
    └── workflows/
        └── llmops.yml        # ★ GitHub Actions CI/CDパイプライン
```

---

## 1. プロンプトのバージョン管理 — `llmops/prompt_registry.py`

プロンプトはコードです。バージョン管理、差分確認、承認ワークフロー、ロールバック機能が必要です。

```python
# llmops/prompt_registry.py
"""
プロンプトのバージョン管理

プロンプトを変更するとRAGの回答品質が大きく変わります。
どのバージョンのプロンプトを使っているかを追跡・管理します。
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PromptVersion:
    version: str           # "v1.0.0"
    name: str              # "rag_answer_prompt"
    template: str          # プロンプトテンプレート
    description: str       # 変更内容の説明
    created_at: str        # 作成日時
    hash: str              # テンプレートのハッシュ値


REGISTRY_FILE = "llmops/prompt_versions.json"

# ── プロンプトテンプレートの定義 ──────────────────────────────
PROMPTS = {
    "rag_answer": {
        "v1.0.0": {
            "template": """以下のドキュメントを参考に、質問に答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答""",
            "description": "初期バージョン",
        },
        "v1.1.0": {
            "template": """以下のドキュメントを参考に、質問に答えてください。
ドキュメントに記載がない場合は「ドキュメントに記載がありません」と答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答（簡潔に、ドキュメントに基づいて）""",
            "description": "ハルシネーション対策：ドキュメント外の情報を答えないよう明示",
        },
        "v1.2.0": {
            "template": """あなたはドキュメント検索アシスタントです。
以下のドキュメントの内容のみに基づいて、質問に答えてください。

制約:
- ドキュメントに記載されていない情報は答えない
- 推測や補完をしない
- 不明な場合は「ドキュメントに記載がありません」と答える

# 参考ドキュメント
{context}

# 質問
{question}

# 回答""",
            "description": "セキュリティ強化：役割・制約を明示したシステムプロンプト形式",
        },
    }
}


def get_prompt(name: str, version: str = "latest") -> str:
    """プロンプトテンプレートを取得する"""
    if name not in PROMPTS:
        raise ValueError(f"プロンプト '{name}' が見つかりません")

    versions = PROMPTS[name]

    if version == "latest":
        version = sorted(versions.keys())[-1]

    if version not in versions:
        raise ValueError(f"バージョン '{version}' が見つかりません")

    return versions[version]["template"]


def list_versions(name: str) -> list[dict]:
    """プロンプトのバージョン一覧を返す"""
    if name not in PROMPTS:
        raise ValueError(f"プロンプト '{name}' が見つかりません")

    result = []
    for version, info in PROMPTS[name].items():
        template_hash = hashlib.md5(info["template"].encode()).hexdigest()[:8]
        result.append({
            "version": version,
            "description": info["description"],
            "hash": template_hash,
        })
    return result


def compare_versions(name: str, v1: str, v2: str) -> dict:
    """2つのバージョンの差分を比較する"""
    t1 = get_prompt(name, v1)
    t2 = get_prompt(name, v2)

    lines1 = set(t1.split("\n"))
    lines2 = set(t2.split("\n"))

    added = lines2 - lines1
    removed = lines1 - lines2

    return {
        "added_lines": len(added),
        "removed_lines": len(removed),
        "char_diff": len(t2) - len(t1),
        "sample_added": list(added)[:3],
    }


# ── 実行 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== プロンプトバージョン一覧 ===\n")
    for version_info in list_versions("rag_answer"):
        print(f"  {version_info['version']} [{version_info['hash']}] - {version_info['description']}")

    print("\n=== v1.0.0 → v1.2.0 の差分 ===")
    diff = compare_versions("rag_answer", "v1.0.0", "v1.2.0")
    print(f"  追加行: {diff['added_lines']}行")
    print(f"  削除行: {diff['removed_lines']}行")
    print(f"  文字数変化: {diff['char_diff']:+d}文字")

    print("\n=== 最新バージョン（v1.2.0）のプロンプト ===")
    print(get_prompt("rag_answer", "latest"))
```

```bash
mkdir llmops
python llmops/prompt_registry.py
```

---

## 2. CI用の評価スクリプト — `llmops/ci_eval.py`

GitHubにpushするたびに自動実行されるEvalスクリプトです。品質基準を下回ったらCIを失敗させます。

```python
# llmops/ci_eval.py
"""
CI/CD用の評価スクリプト

GitHubにpushするたびに実行され、
品質基準を下回った場合はexit code 1を返してCIを失敗させます。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json
from llmops.prompt_registry import get_prompt
from evals.dataset import EVAL_DATASET

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

# ── 品質基準（これを下回ったらCIが失敗する） ─────────────────
QUALITY_THRESHOLDS = {
    "context_recall": 0.80,    # 検索精度80%以上
    "answer_relevancy": 0.70,  # 回答関連性70%以上
    "overall": 0.75,           # 総合スコア75%以上
}

# 評価するプロンプトのバージョン
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "latest")


def get_embedding(text: str) -> list[float]:
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
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "similarity": round(r[2], 4)}
        for r in rows
    ]


def rag_answer_with_prompt(question: str, prompt_version: str) -> tuple[str, list[dict]]:
    """指定バージョンのプロンプトでRAGを実行"""
    docs = search(question, top_k=3)
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    # プロンプトレジストリからテンプレートを取得
    prompt_template = get_prompt("rag_answer", prompt_version)
    prompt = prompt_template.format(context=context, question=question)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text, docs
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 15)
            else:
                raise


def eval_context_recall(retrieved_docs: list[dict], expected_docs: list[str]) -> float:
    retrieved_titles = [d["title"] for d in retrieved_docs]
    hit = sum(1 for expected in expected_docs if expected in retrieved_titles)
    return hit / len(expected_docs) if expected_docs else 0.0


def eval_answer_relevancy(answer: str, keywords: list[str]) -> float:
    hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hit / len(keywords) if keywords else 0.0


def run_ci_eval(prompt_version: str = "latest") -> dict:
    """CI用の評価を実行してレポートを返す"""
    print(f"CI評価開始: プロンプトバージョン={prompt_version}")
    print("=" * 60)

    results = []

    for item in EVAL_DATASET:
        print(f"\n[{item['id']}] {item['question']}")

        try:
            answer, retrieved_docs = rag_answer_with_prompt(item["question"], prompt_version)
            time.sleep(3)

            context_recall = eval_context_recall(retrieved_docs, item["expected_docs"])
            answer_relevancy = eval_answer_relevancy(answer, item["expected_answer_keywords"])
            overall = (context_recall + answer_relevancy) / 2

            results.append({
                "id": item["id"],
                "context_recall": context_recall,
                "answer_relevancy": answer_relevancy,
                "overall": overall,
            })

            status = "✓" if overall >= QUALITY_THRESHOLDS["overall"] else "✗"
            print(f"  {status} Context Recall: {context_recall:.2f} | Relevancy: {answer_relevancy:.2f} | Overall: {overall:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": item["id"],
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
                "overall": 0.0,
                "error": str(e),
            })

    # 集計
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_overall = sum(r["overall"] for r in results) / len(results)

    report = {
        "prompt_version": prompt_version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {
            "context_recall": round(avg_recall, 3),
            "answer_relevancy": round(avg_relevancy, 3),
            "overall": round(avg_overall, 3),
        },
        "thresholds": QUALITY_THRESHOLDS,
        "passed": (
            avg_recall >= QUALITY_THRESHOLDS["context_recall"] and
            avg_relevancy >= QUALITY_THRESHOLDS["answer_relevancy"] and
            avg_overall >= QUALITY_THRESHOLDS["overall"]
        ),
        "results": results,
    }

    return report


if __name__ == "__main__":
    report = run_ci_eval(PROMPT_VERSION)

    print("\n" + "=" * 60)
    print("CI評価レポート")
    print("=" * 60)
    print(f"プロンプトバージョン: {report['prompt_version']}")
    print(f"Context Recall:   {report['metrics']['context_recall']:.3f} (基準: {QUALITY_THRESHOLDS['context_recall']})")
    print(f"Answer Relevancy: {report['metrics']['answer_relevancy']:.3f} (基準: {QUALITY_THRESHOLDS['answer_relevancy']})")
    print(f"Overall:          {report['metrics']['overall']:.3f} (基準: {QUALITY_THRESHOLDS['overall']})")

    # レポートをJSONで保存（GitHub Actionsのアーティファクトとして保存可能）
    with open("llmops/eval_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nレポートを llmops/eval_report.json に保存しました")

    if report["passed"]:
        print("\n✅ CI評価: PASSED — デプロイ可能")
        sys.exit(0)
    else:
        print("\n❌ CI評価: FAILED — 品質基準を満たしていません")
        sys.exit(1)  # CIを失敗させる
```

```bash
python llmops/ci_eval.py
```

---

## 3. APIコスト追跡 — `llmops/cost_tracker.py`

トークンコスト管理はLLMOpsの重要な懸念事項です。LLM APIコールは使用量課金であり、コスト消費の監視と最適化が必要です。

```python
# llmops/cost_tracker.py
"""
APIコスト追跡

Gemini APIの使用量とコストを記録・集計します。
無料枠の残量を把握して上限に近づいたらアラートを出します。
"""
import json
import time
from datetime import datetime, date
from pathlib import Path


COST_LOG_FILE = "llmops/cost_log.json"

# Gemini APIの料金（2026年6月時点の無料枠目安）
PRICING = {
    "gemini-2.5-flash": {
        "input_per_1k_tokens": 0.0,   # 無料枠内
        "output_per_1k_tokens": 0.0,  # 無料枠内
        "free_tier_requests_per_day": 20,
    },
    "gemini-embedding-001": {
        "input_per_1k_tokens": 0.0,
        "free_tier_requests_per_day": 1500,
    },
}


def load_cost_log() -> dict:
    """コストログを読み込む"""
    if Path(COST_LOG_FILE).exists():
        with open(COST_LOG_FILE, "r") as f:
            return json.load(f)
    return {"daily": {}, "total": {"requests": 0, "estimated_cost_usd": 0.0}}


def save_cost_log(log: dict):
    """コストログを保存する"""
    Path(COST_LOG_FILE).parent.mkdir(exist_ok=True)
    with open(COST_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def record_request(model: str, input_tokens: int = 0, output_tokens: int = 0):
    """APIリクエストを記録する"""
    log = load_cost_log()
    today = date.today().isoformat()

    if today not in log["daily"]:
        log["daily"][today] = {}

    if model not in log["daily"][today]:
        log["daily"][today][model] = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    log["daily"][today][model]["requests"] += 1
    log["daily"][today][model]["input_tokens"] += input_tokens
    log["daily"][today][model]["output_tokens"] += output_tokens
    log["total"]["requests"] += 1

    save_cost_log(log)


def get_daily_summary(target_date: str = None) -> dict:
    """日次サマリーを返す"""
    log = load_cost_log()
    target = target_date or date.today().isoformat()

    if target not in log["daily"]:
        return {"date": target, "models": {}, "total_requests": 0, "warnings": []}

    daily = log["daily"][target]
    warnings = []

    for model, stats in daily.items():
        if model in PRICING:
            limit = PRICING[model].get("free_tier_requests_per_day", 0)
            if limit > 0 and stats["requests"] >= limit * 0.8:
                warnings.append(f"{model}: 無料枠の{stats['requests']}/{limit}リクエスト使用（{stats['requests']/limit*100:.0f}%）")

    return {
        "date": target,
        "models": daily,
        "total_requests": sum(s["requests"] for s in daily.values()),
        "warnings": warnings,
    }


def print_cost_report():
    """コストレポートを表示する"""
    summary = get_daily_summary()
    print(f"\n=== APIコストレポート ({summary['date']}) ===\n")

    if not summary["models"]:
        print("  本日のAPIリクエスト記録なし")
        return

    for model, stats in summary["models"].items():
        limit = PRICING.get(model, {}).get("free_tier_requests_per_day", "不明")
        print(f"  {model}:")
        print(f"    リクエスト数: {stats['requests']} / {limit}")
        if stats.get("input_tokens"):
            print(f"    入力トークン: {stats['input_tokens']:,}")
        if stats.get("output_tokens"):
            print(f"    出力トークン: {stats['output_tokens']:,}")

    print(f"\n  合計リクエスト: {summary['total_requests']}")

    if summary["warnings"]:
        print("\n  ⚠️  警告:")
        for warning in summary["warnings"]:
            print(f"    - {warning}")
    else:
        print("\n  ✅ 無料枠内で運用中")


if __name__ == "__main__":
    # テスト用のダミーデータを記録
    record_request("gemini-2.5-flash", input_tokens=150, output_tokens=80)
    record_request("gemini-2.5-flash", input_tokens=200, output_tokens=120)
    record_request("gemini-embedding-001", input_tokens=50)
    record_request("gemini-embedding-001", input_tokens=50)
    record_request("gemini-embedding-001", input_tokens=50)

    print_cost_report()
```

```bash
python llmops/cost_tracker.py
```

---

## 4. GitHub Actions CI/CDパイプライン — `.github/workflows/llmops.yml`

GitHubにpushするたびに自動で評価が走るパイプラインです。

```yaml
# .github/workflows/llmops.yml
name: LLMOps CI/CD Pipeline

on:
  push:
    branches: [main]
    paths:
      - "*.py"
      - "llmops/**"
      - "evals/**"
      - "security/**"
  pull_request:
    branches: [main]

jobs:
  # ── Job 1: 入力検証テスト ────────────────────────────────────
  security-test:
    name: Security Validation
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run input validator tests
        run: |
          python security/input_validator.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}

  # ── Job 2: RAG品質評価（Evalsゲート） ───────────────────────
  eval-gate:
    name: RAG Quality Gate
    runs-on: ubuntu-latest
    needs: security-test  # セキュリティテストが通った後に実行

    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: password
          POSTGRES_DB: vectordb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Setup DB and seed data
        run: |
          python 01_setup_db.py
          python 02_create_index.py
          python 03_ingest.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          DB_HOST: localhost
          DB_PORT: 5432
          DB_NAME: vectordb
          DB_USER: postgres
          DB_PASSWORD: password

      - name: Run CI Eval (Quality Gate)
        run: |
          python llmops/ci_eval.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          DB_HOST: localhost
          DB_PORT: 5432
          DB_NAME: vectordb
          DB_USER: postgres
          DB_PASSWORD: password
          PROMPT_VERSION: latest

      - name: Upload eval report
        uses: actions/upload-artifact@v4
        if: always()  # 失敗してもレポートをアップロード
        with:
          name: eval-report
          path: llmops/eval_report.json

  # ── Job 3: デプロイ（品質ゲートを通過した場合のみ） ─────────
  deploy:
    name: Deploy to Render
    runs-on: ubuntu-latest
    needs: eval-gate  # 評価ゲートが通った後にデプロイ
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: Deploy to Render
        run: |
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
        # Renderのデプロイフックは Render Dashboard → Settings → Deploy Hook で取得
```

---

## 5. GitHub Secretsの設定

### 5-1. Secretsの登録場所

> **注意:** GitHubの「Settings」はアカウント全体の設定と、リポジトリごとの設定の2種類があります。Secretsはリポジトリごとの設定にあります。

1. `github.com/qameqame/pgvector-tutorial` にアクセス（アカウント設定ではなくリポジトリのページ）
2. リポジトリの「**Settings**」タブをクリック
3. 左サイドバーの「**Secrets and variables**」→「**Actions**」
4. 「**New repository secret**」をクリック

以下の2つを追加：

| Secret名 | 値 |
|---------|-----|
| `GEMINI_API_KEY` | `AIza...`（Gemini APIキー） |
| `RENDER_DEPLOY_HOOK_URL` | RenderのDeploy Hook URL |

### 5-2. RenderのDeploy Hook URLの取得方法

1. [dashboard.render.com](https://dashboard.render.com) にアクセス
2. `pgvector-mcp-server` のサービスをクリック
3. 「**Settings**」タブをクリック
4. ページを下にスクロールして「**Deploy Hook**」セクションを探す
5. 「**Generate Deploy Hook**」をクリック
6. 表示されたURLをコピー（`https://api.render.com/deploy/srv-...?key=...` の形式）

---

## 6. LLMOpsの全体像

今回の実装をまとめると、以下のパイプラインが完成します：

```
開発者がコードをpush
    ↓
GitHub Actions起動
    ↓
[Job 1] セキュリティテスト
  → input_validator.py でプロンプトインジェクション検知テスト
    ↓ 成功
[Job 2] RAG品質評価（Evalsゲート）
  → ci_eval.py でContext Recall・Answer Relevancyを自動測定
  → 品質基準（Overall 75%以上）を満たしているか確認
  → 評価レポートをアーティファクトとして保存
    ↓ 成功
[Job 3] 本番デプロイ
  → Renderに自動デプロイ
  → Langfuseでトレーシング開始
```

---

## 7. GitHub Actionsの動作確認

### 7-1. 手動でトリガーする

Secretsを設定したらpushしてActionsを起動します：

```bash
cd /Users/kameyama/dev/ai-agent/pgvector-tutorial
git commit --allow-empty -m "ci: trigger GitHub Actions test"
git push origin main
```

### 7-2. 実行状況の確認

1. `github.com/qameqame/pgvector-tutorial` にアクセス
2. 「**Actions**」タブをクリック
3. ワークフローの実行履歴が表示されます

| アイコン | 意味 |
|---------|------|
| 🟡 黄色の丸 | 実行中 |
| ✅ 緑のチェック | 成功（デプロイまで完了） |
| ❌ 赤のバツ | 失敗（ログを確認） |

### 7-3. 失敗時のログ確認

失敗した場合はワークフロー名をクリック → 失敗したJobをクリック → 各ステップのログを確認します。

よくある失敗パターン：

| 失敗したJob | 原因 | 対処 |
|------------|------|------|
| Security Validation | `GEMINI_API_KEY` が未設定 | Secretsを確認 |
| RAG Quality Gate | DBへの接続失敗 | `services.postgres` の設定を確認 |
| Deploy to Render | `RENDER_DEPLOY_HOOK_URL` が未設定 | Secretsを確認 |

### 7-4. 評価レポートのダウンロード

CI評価のレポートはアーティファクトとして保存されます。

1. ワークフローの実行ページを開く
2. 下部の「**Artifacts**」セクション
3. `eval-report` をクリックしてダウンロード
4. `llmops/eval_report.json` の内容を確認

プロンプトを変更したとき、どちらが良いかを比較する方法です：

```bash
# v1.1.0で評価
PROMPT_VERSION=v1.1.0 python llmops/ci_eval.py

# v1.2.0で評価
PROMPT_VERSION=v1.2.0 python llmops/ci_eval.py

# スコアを比較してより良いバージョンをlatestに設定
```

---

## よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `ModuleNotFoundError: llmops` | パスが通っていない | `sys.path.append(...)` を確認 |
| CI評価がFAILED | スコアが基準を下回っている | プロンプトを改善 or 基準値を調整 |
| GitHub Actionsがタイムアウト | Geminiのレート制限 | `time.sleep()` を増やす |
| Render Deploy Hookが動かない | Secret未設定 | GitHub Secretsを確認 |

---

## 次のステップ

- **[第6章 Fine-tuning](./06_finetuning)** — LoRAで自社データにモデルを特化させる
- **マルチエージェント** — 複数のAgentが協調するシステムの設計
- **ガバナンス** — EU AI Actへの対応・リスク管理・監査ログ

---

## 参考

- [LangSmith — LLMOpsプラットフォーム](https://smith.langchain.com)
- [MLflow — 実験管理・モデルレジストリ](https://mlflow.org)
- [GitHub Actions 公式ドキュメント](https://docs.github.com/actions)
- [Render Deploy Hooks](https://render.com/docs/deploy-hooks)
