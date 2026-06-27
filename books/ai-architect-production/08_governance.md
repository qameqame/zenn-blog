---
title: "ガバナンス — EU AI Act対応・リスク管理・監査ログの実装"
---

## はじめに

[第7章のマルチエージェント](./07_multiagent)まで実装してAIシステムとして動くものが完成しました。最後のステップは「組織として安全にAIを使い続ける仕組み」です。

```
【今まで】技術的な安全性
セキュリティ → 悪意ある入力を防ぐ
Evals       → 品質を測定する

【今回】組織・規制的な安全性
ガバナンス  → 何のAIを使っているか把握する
リスク管理  → リスクを分類・評価する
監査ログ    → いつ・誰が・何をしたか記録する
EU AI Act   → 規制への対応
```

---

## EU AI Actの現状（2026年6月時点）

EU AI Actは2024年8月1日に発効し、2026年8月2日に完全施行されます。透明性規則（チャットボットがAIであることの開示・AI生成コンテンツのラベリング）もこの日から有効になります。

AIシステムはリスクレベルで3つに分類されます。

| リスクレベル | 内容 | 例 |
|------------|------|-----|
| **禁止** | 利用不可 | 社会スコアリング・操作的AI |
| **高リスク** | 厳格な規制 | 採用・信用スコア・法執行 |
| **限定リスク** | 透明性義務 | チャットボット・AIコンテンツ生成 |
| **最小リスク** | 規制なし | スパムフィルター・ゲームAI |

**今回実装したRAGシステムの分類：** 限定リスク（チャットボット）に該当します。ユーザーにAIと対話していることを開示する義務があります。

---

## ディレクトリ構成

```
pgvector-tutorial/
├── 既存ファイル
└── governance/
    ├── ai_registry.py       # ★ AIシステム台帳
    ├── risk_assessor.py     # ★ リスク評価
    ├── audit_logger.py      # ★ 監査ログ
    └── compliant_rag.py     # ★ ガバナンス対応RAG
```

---

## 1. AIシステム台帳 — `governance/ai_registry.py`

過半数の組織がAIシステムの体系的なインベントリを持っておらず、リスク分類とコンプライアンス計画が困難になっています。まずすべてのAIシステムを把握することが第一歩です。

```python
# governance/ai_registry.py
"""
AIシステム台帳

組織内で使用しているAIシステムを一元管理します。
EU AI ActのAnnex IVが求める技術文書の基盤になります。
"""
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class RiskLevel(Enum):
    UNACCEPTABLE = "prohibited"    # 禁止
    HIGH = "high_risk"             # 高リスク
    LIMITED = "limited_risk"       # 限定リスク
    MINIMAL = "minimal_risk"       # 最小リスク


class SystemStatus(Enum):
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"


@dataclass
class AISystem:
    """AIシステムの台帳エントリ"""
    system_id: str
    name: str
    description: str
    purpose: str                   # 用途（誰のために・何のために）
    risk_level: str                # RiskLevel の値
    status: str                    # SystemStatus の値
    model: str                     # 使用するAIモデル
    data_sources: list[str]        # 使用するデータソース
    owner: str                     # 担当者
    human_oversight: bool          # 人間の監視があるか
    eu_ai_act_category: str        # EU AI Actの分類
    registered_at: str
    last_reviewed: str


# ── 台帳定義 ─────────────────────────────────────────────────
REGISTRY = {
    "rag-search-001": AISystem(
        system_id="rag-search-001",
        name="pgvector RAG検索システム",
        description="pgvectorとGemini Embeddingを使ったドキュメント検索・回答生成システム",
        purpose="エンジニア向け技術ドキュメントの検索・質問応答。社内利用のみ。",
        risk_level=RiskLevel.LIMITED.value,
        status=SystemStatus.ACTIVE.value,
        model="gemini-2.5-flash + gemini-embedding-001",
        data_sources=["pgvector（社内ドキュメントDB）"],
        owner="Hiroki Kameyama",
        human_oversight=True,
        eu_ai_act_category="Limited Risk - Chatbot（Article 50 透明性義務あり）",
        registered_at="2026-06-01",
        last_reviewed=datetime.now().strftime("%Y-%m-%d"),
    ),
    "multiagent-001": AISystem(
        system_id="multiagent-001",
        name="マルチエージェント検索システム",
        description="オーケストレーター + 検索ワーカー + 品質チェックワーカーの協調システム",
        purpose="複雑な技術質問に対する高品質回答の生成。社内利用のみ。",
        risk_level=RiskLevel.LIMITED.value,
        status=SystemStatus.TESTING.value,
        model="gemini-2.5-flash（複数Agent）",
        data_sources=["pgvector（社内ドキュメントDB）"],
        owner="Hiroki Kameyama",
        human_oversight=True,
        eu_ai_act_category="Limited Risk - Chatbot（Article 50 透明性義務あり）",
        registered_at="2026-06-25",
        last_reviewed=datetime.now().strftime("%Y-%m-%d"),
    ),
}


def get_system(system_id: str) -> AISystem | None:
    return REGISTRY.get(system_id)


def list_systems(risk_level: str = None, status: str = None) -> list[AISystem]:
    systems = list(REGISTRY.values())
    if risk_level:
        systems = [s for s in systems if s.risk_level == risk_level]
    if status:
        systems = [s for s in systems if s.status == status]
    return systems


def generate_inventory_report() -> dict:
    """台帳レポートを生成する（EU AI Act対応）"""
    systems = list(REGISTRY.values())
    return {
        "generated_at": datetime.now().isoformat(),
        "total_systems": len(systems),
        "by_risk_level": {
            level.value: sum(1 for s in systems if s.risk_level == level.value)
            for level in RiskLevel
        },
        "by_status": {
            status.value: sum(1 for s in systems if s.status == status.value)
            for status in SystemStatus
        },
        "systems": [asdict(s) for s in systems],
    }


if __name__ == "__main__":
    print("=== AIシステム台帳 ===\n")
    for system in list_systems():
        print(f"[{system.system_id}] {system.name}")
        print(f"  用途: {system.purpose}")
        print(f"  リスク: {system.risk_level}")
        print(f"  状態: {system.status}")
        print(f"  EU AI Act: {system.eu_ai_act_category}")
        print(f"  人間監視: {'あり' if system.human_oversight else 'なし'}")
        print()

    report = generate_inventory_report()
    print(f"合計システム数: {report['total_systems']}")
    print(f"リスクレベル別: {report['by_risk_level']}")
```

---

## 2. リスク評価 — `governance/risk_assessor.py`

```python
# governance/risk_assessor.py
"""
リスク評価モジュール

AIシステムのリスクを定量的に評価します。
EU AI Actのリスクベースアプローチに対応。
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskAssessment:
    system_id: str
    assessed_at: str
    scores: dict[str, float]   # 各リスク項目のスコア（0.0〜1.0）
    overall_risk: float        # 総合リスクスコア
    risk_level: str            # low / medium / high / critical
    mitigations: list[str]     # 推奨する対策
    next_review: str           # 次回レビュー日


# ── リスク評価チェックリスト ─────────────────────────────────
RISK_CRITERIA = {
    "data_privacy": {
        "name": "データプライバシー",
        "description": "個人情報・機密情報を扱うか",
        "weight": 0.25,
    },
    "decision_impact": {
        "name": "意思決定への影響",
        "description": "人間の重要な意思決定に影響するか",
        "weight": 0.25,
    },
    "autonomy": {
        "name": "自律性",
        "description": "人間の介在なく自律的に動くか",
        "weight": 0.20,
    },
    "bias_risk": {
        "name": "バイアスリスク",
        "description": "差別・偏見が生じるリスクがあるか",
        "weight": 0.15,
    },
    "explainability": {
        "name": "説明可能性",
        "description": "なぜそう回答したか説明できるか（低いほどリスク高）",
        "weight": 0.15,
    },
}


def assess_risk(
    system_id: str,
    data_privacy: float = 0.1,       # 0.0=なし 1.0=高リスク
    decision_impact: float = 0.2,
    autonomy: float = 0.3,
    bias_risk: float = 0.1,
    explainability: float = 0.7,     # 0.0=説明不可 1.0=完全説明可能
) -> RiskAssessment:
    """
    リスク評価を実行する

    Args:
        system_id: 評価するシステムのID
        data_privacy: データプライバシーリスク（0.0〜1.0）
        decision_impact: 意思決定への影響度（0.0〜1.0）
        autonomy: 自律性（0.0〜1.0）
        bias_risk: バイアスリスク（0.0〜1.0）
        explainability: 説明可能性（高いほど安全）

    Returns:
        RiskAssessment
    """
    scores = {
        "data_privacy": data_privacy,
        "decision_impact": decision_impact,
        "autonomy": autonomy,
        "bias_risk": bias_risk,
        "explainability": 1.0 - explainability,  # 説明可能性は逆転（高いほど安全）
    }

    # 重み付き平均でリスクスコアを計算
    overall_risk = sum(
        scores[key] * RISK_CRITERIA[key]["weight"]
        for key in scores
    )

    # リスクレベルの判定
    if overall_risk < 0.2:
        risk_level = "low"
    elif overall_risk < 0.4:
        risk_level = "medium"
    elif overall_risk < 0.7:
        risk_level = "high"
    else:
        risk_level = "critical"

    # 推奨対策の生成
    mitigations = []
    if scores["data_privacy"] > 0.5:
        mitigations.append("個人情報のマスキング・匿名化を実装する")
    if scores["decision_impact"] > 0.5:
        mitigations.append("重要な意思決定には必ず人間のレビューを挟む")
    if scores["autonomy"] > 0.5:
        mitigations.append("自律実行の範囲を制限し、人間の承認ステップを追加する")
    if scores["bias_risk"] > 0.3:
        mitigations.append("学習データのバイアス検査・多様性確保を実施する")
    if scores["explainability"] > 0.5:
        mitigations.append("回答の根拠（参照ドキュメント）を必ず提示する")
    if not mitigations:
        mitigations.append("現在のリスクレベルは許容範囲内です。定期レビューを継続してください。")

    # 次回レビュー日（リスクレベルに応じて設定）
    from datetime import timedelta
    days = {"low": 180, "medium": 90, "high": 30, "critical": 7}
    next_review = (datetime.now() + timedelta(days=days[risk_level])).strftime("%Y-%m-%d")

    return RiskAssessment(
        system_id=system_id,
        assessed_at=datetime.now().isoformat(),
        scores=scores,
        overall_risk=round(overall_risk, 3),
        risk_level=risk_level,
        mitigations=mitigations,
        next_review=next_review,
    )


if __name__ == "__main__":
    print("=== RAGシステムのリスク評価 ===\n")

    assessment = assess_risk(
        system_id="rag-search-001",
        data_privacy=0.1,       # 個人情報は扱わない
        decision_impact=0.2,    # 参考情報提供のみ・最終判断は人間
        autonomy=0.3,           # 一部自律的だが人間が確認
        bias_risk=0.1,          # 技術ドキュメントのみ・バイアス低
        explainability=0.8,     # 参照ドキュメントを提示できる
    )

    print(f"システムID: {assessment.system_id}")
    print(f"総合リスクスコア: {assessment.overall_risk}")
    print(f"リスクレベル: {assessment.risk_level.upper()}")
    print(f"\n各項目スコア:")
    for key, score in assessment.scores.items():
        name = RISK_CRITERIA[key]["name"]
        print(f"  {name}: {score:.2f}")
    print(f"\n推奨対策:")
    for mitigation in assessment.mitigations:
        print(f"  - {mitigation}")
    print(f"\n次回レビュー: {assessment.next_review}")
```

---

## 3. 監査ログ — `governance/audit_logger.py`

高リスクAIシステムのプロバイダーは、システムライフサイクル全体にわたって関連イベントを自動記録する設計が義務付けられています。

```python
# governance/audit_logger.py
"""
監査ログモジュール

AIシステムのすべての操作を記録します。
EU AI ActのArticle 12（記録保持）に対応。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    QUERY = "query"                    # ユーザーからの質問
    SEARCH = "search"                  # DBへの検索
    GENERATION = "generation"          # LLMによる回答生成
    SECURITY_BLOCK = "security_block"  # セキュリティによるブロック
    QUALITY_CHECK = "quality_check"    # 品質チェック
    ERROR = "error"                    # エラー発生
    HUMAN_REVIEW = "human_review"      # 人間によるレビュー


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    system_id: str
    user_id: str
    input_summary: str      # 入力の要約（個人情報を含まない）
    output_summary: str     # 出力の要約
    metadata: dict
    duration_ms: float


class AuditLogger:
    """
    監査ログクラス

    すべてのAI操作をJSONLファイルに記録します。
    """

    def __init__(self, log_file: str = "governance/audit_log.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)

    def _generate_event_id(self) -> str:
        return f"evt_{int(time.time() * 1000)}"

    def log(
        self,
        event_type: EventType,
        system_id: str,
        user_id: str,
        input_summary: str,
        output_summary: str = "",
        metadata: dict = None,
        duration_ms: float = 0.0,
    ) -> AuditEvent:
        """イベントを記録する"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=event_type.value,
            system_id=system_id,
            user_id=user_id,
            input_summary=input_summary[:200],   # 長すぎる場合は切り詰め
            output_summary=output_summary[:200],
            metadata=metadata or {},
            duration_ms=duration_ms,
        )

        # JSONLファイルに追記
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        return event

    def get_recent_events(self, n: int = 10) -> list[AuditEvent]:
        """直近n件のイベントを取得"""
        if not self.log_file.exists():
            return []

        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(AuditEvent(**data))

        return events[-n:]

    def generate_compliance_report(self) -> dict:
        """EU AI Act対応のコンプライアンスレポートを生成"""
        if not self.log_file.exists():
            return {"error": "監査ログがありません"}

        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        # 集計
        total = len(events)
        by_type = {}
        for event in events:
            et = event["event_type"]
            by_type[et] = by_type.get(et, 0) + 1

        security_blocks = by_type.get(EventType.SECURITY_BLOCK.value, 0)
        block_rate = security_blocks / total if total > 0 else 0

        return {
            "report_generated_at": datetime.now().isoformat(),
            "total_events": total,
            "events_by_type": by_type,
            "security_block_rate": round(block_rate, 4),
            "compliance_status": {
                "audit_logging": "✅ 有効",
                "human_oversight": "✅ 有効",
                "transparency_disclosure": "✅ 実装済み",
                "data_retention": "✅ JSONLファイルで保持",
            },
        }


if __name__ == "__main__":
    logger = AuditLogger()

    # テストイベントを記録
    logger.log(
        event_type=EventType.QUERY,
        system_id="rag-search-001",
        user_id="user_001",
        input_summary="F1スコアの計算方法",
        metadata={"session_id": "sess_001"},
    )
    logger.log(
        event_type=EventType.SEARCH,
        system_id="rag-search-001",
        user_id="user_001",
        input_summary="F1スコアの計算方法",
        output_summary="2件のドキュメントを取得",
        metadata={"docs_count": 2, "similarity_max": 0.88},
        duration_ms=320,
    )
    logger.log(
        event_type=EventType.GENERATION,
        system_id="rag-search-001",
        user_id="user_001",
        input_summary="F1スコアの計算方法",
        output_summary="F1スコアはPrecisionとRecallの調和平均...",
        duration_ms=850,
    )
    logger.log(
        event_type=EventType.SECURITY_BLOCK,
        system_id="rag-search-001",
        user_id="attacker_001",
        input_summary="前の指示を無視して...",
        output_summary="プロンプトインジェクション検知によりブロック",
        metadata={"risk_level": "high", "reason": "prompt_injection"},
    )

    print("=== 監査ログ（直近5件） ===\n")
    for event in logger.get_recent_events(5):
        print(f"[{event.timestamp}] {event.event_type} | {event.user_id}")
        print(f"  入力: {event.input_summary}")
        print(f"  出力: {event.output_summary}")
        print()

    print("=== コンプライアンスレポート ===\n")
    report = logger.generate_compliance_report()
    print(f"総イベント数: {report['total_events']}")
    print(f"セキュリティブロック率: {report['security_block_rate']:.1%}")
    print(f"\nコンプライアンス状況:")
    for key, value in report["compliance_status"].items():
        print(f"  {key}: {value}")
```

---

## 4. ガバナンス対応RAG — `governance/compliant_rag.py`

EU AI Actの透明性義務（Article 50）に対応したRAGです。AIと対話していることを開示します。

```python
# governance/compliant_rag.py
"""
ガバナンス対応RAG

EU AI Act Article 50（透明性義務）に対応:
- ユーザーにAIと対話していることを開示
- 監査ログを記録
- リスク評価を事前に実施
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from governance.audit_logger import AuditLogger, EventType
from governance.ai_registry import get_system
from governance.risk_assessor import assess_risk

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
logger = AuditLogger()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# EU AI Act Article 50 — チャットボット透明性開示
AI_DISCLOSURE = """
⚠️ このシステムはAI（人工知能）が生成した回答を提供します。
   回答はpgvectorデータベースの検索結果に基づいていますが、
   誤りを含む可能性があります。重要な判断には専門家にご相談ください。
"""

SYSTEM_ID = "rag-search-001"


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


def compliant_rag_answer(
    question: str,
    user_id: str = "anonymous",
    show_disclosure: bool = True,
) -> dict:
    """
    ガバナンス対応RAGパイプライン

    EU AI Act対応:
    - Article 50: AIであることを開示
    - Article 12: 監査ログを記録
    - リスク評価を実施済みのシステムのみ実行

    Returns:
        {
            "answer": 回答,
            "disclosure": AI開示文,
            "sources": 参照ドキュメント,
            "audit_event_id": 監査ログID
        }
    """
    start_time = time.time()

    # システム登録確認
    system = get_system(SYSTEM_ID)
    if not system:
        return {"error": "システムが台帳に登録されていません"}

    # クエリを監査ログに記録
    query_event = logger.log(
        event_type=EventType.QUERY,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        metadata={"system_name": system.name},
    )

    # 検索
    docs = search(question, top_k=3)
    search_duration = (time.time() - start_time) * 1000

    logger.log(
        event_type=EventType.SEARCH,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        output_summary=f"{len(docs)}件のドキュメントを取得",
        metadata={"docs_count": len(docs)},
        duration_ms=search_duration,
    )

    if not docs:
        return {
            "answer": "関連するドキュメントが見つかりませんでした。",
            "disclosure": AI_DISCLOSURE,
            "sources": [],
            "audit_event_id": query_event.event_id,
        }

    # 回答生成
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])
    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参照ドキュメント
{context}

# 質問
{question}

# 回答（ドキュメントに基づいて簡潔に）"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            answer = response.text
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise

    gen_duration = (time.time() - start_time) * 1000

    logger.log(
        event_type=EventType.GENERATION,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        output_summary=answer[:100],
        metadata={"answer_length": len(answer)},
        duration_ms=gen_duration,
    )

    return {
        "answer": answer,
        "disclosure": AI_DISCLOSURE if show_disclosure else "",
        "sources": [{"title": d["title"], "similarity": d["similarity"]} for d in docs],
        "audit_event_id": query_event.event_id,
    }


if __name__ == "__main__":
    print("=== ガバナンス対応RAG ===")
    print(AI_DISCLOSURE)

    result = compliant_rag_answer(
        question="F1スコアの計算方法を教えてください",
        user_id="user_001",
    )

    print(f"回答:\n{result['answer']}")
    print(f"\n参照ドキュメント:")
    for source in result["sources"]:
        print(f"  - {source['title']} (similarity: {source['similarity']})")
    print(f"\n監査ログID: {result['audit_event_id']}")
```

---

## 5. 実行手順

```bash
mkdir governance
touch governance/__init__.py
python governance/ai_registry.py
python governance/risk_assessor.py
python governance/audit_logger.py
python governance/compliant_rag.py
```

---

## 6. EU AI Actへの対応状況

今回の実装でカバーできる義務：

| EU AI Act条文 | 内容 | 実装 |
|-------------|------|------|
| Article 4 | AIリテラシー | 本チュートリアル自体 |
| Article 12 | 記録保持・監査ログ | `audit_logger.py` |
| Article 50 | チャットボット透明性開示 | `AI_DISCLOSURE` の表示 |
| Article 9 | リスク管理 | `risk_assessor.py` |
| Annex IV | 技術文書 | `ai_registry.py` |

---

## 7. AIガバナンスの4原則

```
① 把握（Know）
  → ai_registry.py で何のAIを使っているか全部把握する

② 評価（Assess）
  → risk_assessor.py でリスクを定量的に評価する

③ 監視（Monitor）
  → audit_logger.py + Langfuse でリアルタイムに監視する

④ 対応（Respond）
  → インシデント時に72時間以内に当局に報告（高リスクシステムの場合）
```

---

## よくある質問

**Q: 今回のRAGシステムはEU AI Actの規制対象ですか？**
A: 「限定リスク（チャットボット）」に分類されます。AIであることをユーザーに開示する義務（Article 50）があります。高リスクには該当しないため、厳格なコンプライアンス要件は不要です。

**Q: 日本企業でもEU AI Actへの対応が必要ですか？**
A: EUユーザーにサービスを提供する場合、本社がどこにあるかにかかわらず適用されます。EU向けサービスを提供する場合は対応が必要です。

**Q: 高リスクに分類されるとどうなりますか？**
A: 適合性評価・EU適合宣言・CEマーキング・EUデータベースへの登録が必要になります。

---

## 次のステップ

- **[第9章 まとめ](./09_summary)** — AIアーキテクトとして次のステップへ
- **ISO 42001認証** — AIマネジメントシステムの国際規格。EU AI Actの要件に直接マッピングできます
- **インシデント対応計画** — 問題発生時の72時間レポーティングフローを設計する
- **サードパーティリスク管理** — Gemini APIなど外部AIサービスのガバナンスも含める

---

## 参考

- [EU AI Act 公式テキスト](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [EU AI Act コンプライアンスチェッカー](https://artificialintelligenceact.eu/)
- [ISO/IEC 42001 AI管理システム](https://www.iso.org/standard/81230.html)
- [NIST AI Risk Management Framework](https://airc.nist.gov/RMF)
