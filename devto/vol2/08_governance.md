---
title: AI Governance — EU AI Act Compliance, Risk Assessment, and Audit Logging
published: false
tags: ai, mlops, llm, python
series: AI Architect's Production Operations Guide
---

## Introduction

Through [Chapter 7 (Multi-Agent)](./07_multiagent), we have a complete, functioning AI system. The final step is building *organizational* infrastructure to operate AI safely over time.

```
[Before] Technical safety
Security → Block malicious input
Evals    → Measure quality

[Now] Organizational / regulatory safety
Governance   → Know what AI systems are in use
Risk mgmt    → Classify and assess risks
Audit logs   → Record who did what, when
EU AI Act    → Regulatory compliance
```

---

## EU AI Act Status (as of June 2026)

The EU AI Act came into force on August 1, 2024, with full enforcement on **August 2, 2026**. Transparency rules (disclosing when users are interacting with AI, labeling AI-generated content) also take effect on that date.

AI systems are classified into three risk tiers:

| Risk Level | Description | Examples |
|------------|-------------|---------|
| **Prohibited** | Not permitted | Social scoring, manipulative AI |
| **High Risk** | Strict regulation | Hiring, credit scoring, law enforcement |
| **Limited Risk** | Transparency obligations | Chatbots, AI-generated content |
| **Minimal Risk** | No regulation | Spam filters, game AI |

**Our RAG system's classification:** Limited Risk (chatbot). We are required to disclose to users that they are interacting with AI.

---

## Directory Structure

```
pgvector-tutorial/
├── existing files
└── governance/
    ├── ai_registry.py       # ★ AI system inventory
    ├── risk_assessor.py     # ★ Risk assessment
    ├── audit_logger.py      # ★ Audit logging
    └── compliant_rag.py     # ★ Governance-compliant RAG
```

---

## 1. AI System Inventory — `governance/ai_registry.py`

Most organizations lack a systematic inventory of their AI systems, making risk classification and compliance planning difficult. Knowing what you have is the essential first step.

```python
# governance/ai_registry.py
"""
AI system inventory

Centrally manage all AI systems in use across the organization.
Forms the foundation for technical documentation required by EU AI Act Annex IV.
"""
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class RiskLevel(Enum):
    UNACCEPTABLE = "prohibited"
    HIGH = "high_risk"
    LIMITED = "limited_risk"
    MINIMAL = "minimal_risk"


class SystemStatus(Enum):
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"


@dataclass
class AISystem:
    system_id: str
    name: str
    description: str
    purpose: str
    risk_level: str
    status: str
    model: str
    data_sources: list[str]
    owner: str
    human_oversight: bool
    eu_ai_act_category: str
    registered_at: str
    last_reviewed: str


REGISTRY = {
    "rag-search-001": AISystem(
        system_id="rag-search-001",
        name="pgvector RAG Search System",
        description="Document search and answer generation using pgvector and Gemini Embedding",
        purpose="Technical document search and Q&A for engineers. Internal use only.",
        risk_level=RiskLevel.LIMITED.value,
        status=SystemStatus.ACTIVE.value,
        model="gemini-2.5-flash + gemini-embedding-001",
        data_sources=["pgvector (internal document DB)"],
        owner="Hiroki Kameyama",
        human_oversight=True,
        eu_ai_act_category="Limited Risk - Chatbot (Article 50 transparency obligations apply)",
        registered_at="2026-06-01",
        last_reviewed=datetime.now().strftime("%Y-%m-%d"),
    ),
    "multiagent-001": AISystem(
        system_id="multiagent-001",
        name="Multi-Agent Search System",
        description="Collaborative system with Orchestrator + Search Worker + Quality Check Worker",
        purpose="High-quality answer generation for complex technical questions. Internal use only.",
        risk_level=RiskLevel.LIMITED.value,
        status=SystemStatus.TESTING.value,
        model="gemini-2.5-flash (multiple Agents)",
        data_sources=["pgvector (internal document DB)"],
        owner="Hiroki Kameyama",
        human_oversight=True,
        eu_ai_act_category="Limited Risk - Chatbot (Article 50 transparency obligations apply)",
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
    """Generate an inventory report (EU AI Act compliant)."""
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
    print("=== AI System Inventory ===\n")
    for system in list_systems():
        print(f"[{system.system_id}] {system.name}")
        print(f"  Purpose: {system.purpose}")
        print(f"  Risk: {system.risk_level}")
        print(f"  Status: {system.status}")
        print(f"  EU AI Act: {system.eu_ai_act_category}")
        print(f"  Human oversight: {'Yes' if system.human_oversight else 'No'}")
        print()

    report = generate_inventory_report()
    print(f"Total systems: {report['total_systems']}")
    print(f"By risk level: {report['by_risk_level']}")
```

---

## 2. Risk Assessment — `governance/risk_assessor.py`

```python
# governance/risk_assessor.py
"""
Risk assessment module

Quantitatively assess AI system risk.
Supports the EU AI Act risk-based approach.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RiskAssessment:
    system_id: str
    assessed_at: str
    scores: dict[str, float]
    overall_risk: float
    risk_level: str          # low / medium / high / critical
    mitigations: list[str]
    next_review: str


RISK_CRITERIA = {
    "data_privacy": {
        "name": "Data Privacy",
        "description": "Does it handle personal or sensitive data?",
        "weight": 0.25,
    },
    "decision_impact": {
        "name": "Decision Impact",
        "description": "Does it influence important human decisions?",
        "weight": 0.25,
    },
    "autonomy": {
        "name": "Autonomy",
        "description": "Does it operate autonomously without human intervention?",
        "weight": 0.20,
    },
    "bias_risk": {
        "name": "Bias Risk",
        "description": "Is there risk of discrimination or bias?",
        "weight": 0.15,
    },
    "explainability": {
        "name": "Explainability",
        "description": "Can it explain why it gave an answer? (lower = higher risk)",
        "weight": 0.15,
    },
}


def assess_risk(
    system_id: str,
    data_privacy: float = 0.1,
    decision_impact: float = 0.2,
    autonomy: float = 0.3,
    bias_risk: float = 0.1,
    explainability: float = 0.7,    # Higher = safer
) -> RiskAssessment:
    """
    Run a risk assessment.

    Args:
        system_id: ID of the system to assess
        data_privacy: Data privacy risk (0.0–1.0)
        decision_impact: Decision influence level (0.0–1.0)
        autonomy: Autonomy level (0.0–1.0)
        bias_risk: Bias risk (0.0–1.0)
        explainability: Explainability (higher = safer, inverted in scoring)
    """
    scores = {
        "data_privacy": data_privacy,
        "decision_impact": decision_impact,
        "autonomy": autonomy,
        "bias_risk": bias_risk,
        "explainability": 1.0 - explainability,  # Invert: high explainability = safer
    }

    overall_risk = sum(
        scores[key] * RISK_CRITERIA[key]["weight"]
        for key in scores
    )

    if overall_risk < 0.2:
        risk_level = "low"
    elif overall_risk < 0.4:
        risk_level = "medium"
    elif overall_risk < 0.7:
        risk_level = "high"
    else:
        risk_level = "critical"

    mitigations = []
    if scores["data_privacy"] > 0.5:
        mitigations.append("Implement PII masking and anonymization")
    if scores["decision_impact"] > 0.5:
        mitigations.append("Require human review for all important decisions")
    if scores["autonomy"] > 0.5:
        mitigations.append("Limit autonomous execution scope, add human approval steps")
    if scores["bias_risk"] > 0.3:
        mitigations.append("Conduct bias audit and ensure diversity in training data")
    if scores["explainability"] > 0.5:
        mitigations.append("Always present the source documents (reference) for answers")
    if not mitigations:
        mitigations.append("Current risk level is within acceptable range. Continue regular reviews.")

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
    print("=== RAG System Risk Assessment ===\n")

    assessment = assess_risk(
        system_id="rag-search-001",
        data_privacy=0.1,       # No personal data handled
        decision_impact=0.2,    # Provides reference info only — humans make final decisions
        autonomy=0.3,           # Partially autonomous but human-reviewed
        bias_risk=0.1,          # Technical docs only — low bias risk
        explainability=0.8,     # Can show reference documents
    )

    print(f"System ID: {assessment.system_id}")
    print(f"Overall risk score: {assessment.overall_risk}")
    print(f"Risk level: {assessment.risk_level.upper()}")
    print(f"\nPer-dimension scores:")
    for key, score in assessment.scores.items():
        name = RISK_CRITERIA[key]["name"]
        print(f"  {name}: {score:.2f}")
    print(f"\nRecommended mitigations:")
    for mitigation in assessment.mitigations:
        print(f"  - {mitigation}")
    print(f"\nNext review: {assessment.next_review}")
```

---

## 3. Audit Logging — `governance/audit_logger.py`

High-risk AI system providers are required to design systems that automatically log relevant events across the system lifecycle.

```python
# governance/audit_logger.py
"""
Audit logging module

Record all operations of AI systems.
Complies with EU AI Act Article 12 (record keeping).
"""
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    QUERY = "query"
    SEARCH = "search"
    GENERATION = "generation"
    SECURITY_BLOCK = "security_block"
    QUALITY_CHECK = "quality_check"
    ERROR = "error"
    HUMAN_REVIEW = "human_review"


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    system_id: str
    user_id: str
    input_summary: str    # Summary of input (no PII)
    output_summary: str   # Summary of output
    metadata: dict
    duration_ms: float


class AuditLogger:
    """
    Audit logger class.
    Records all AI operations to a JSONL file.
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
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=event_type.value,
            system_id=system_id,
            user_id=user_id,
            input_summary=input_summary[:200],
            output_summary=output_summary[:200],
            metadata=metadata or {},
            duration_ms=duration_ms,
        )

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        return event

    def get_recent_events(self, n: int = 10) -> list[AuditEvent]:
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
        """Generate an EU AI Act compliance report."""
        if not self.log_file.exists():
            return {"error": "No audit log found"}

        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

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
                "audit_logging": "✅ Enabled",
                "human_oversight": "✅ Enabled",
                "transparency_disclosure": "✅ Implemented",
                "data_retention": "✅ Retained in JSONL files",
            },
        }
```

---

## 4. Governance-Compliant RAG — `governance/compliant_rag.py`

A RAG implementation compliant with EU AI Act Article 50 (transparency obligations).

```python
# governance/compliant_rag.py
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

# EU AI Act Article 50 — Chatbot transparency disclosure
AI_DISCLOSURE = """
⚠️  This system provides answers generated by Artificial Intelligence.
    Responses are based on pgvector database search results but may contain errors.
    For important decisions, please consult a qualified expert.
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
    Governance-compliant RAG pipeline.

    EU AI Act compliance:
    - Article 50: Disclose that the system is AI
    - Article 12: Record audit logs
    - Only run systems registered in the inventory

    Returns:
        {
            "answer": str,
            "disclosure": str,
            "sources": list,
            "audit_event_id": str
        }
    """
    start_time = time.time()

    system = get_system(SYSTEM_ID)
    if not system:
        return {"error": "System not registered in inventory"}

    query_event = logger.log(
        event_type=EventType.QUERY,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        metadata={"system_name": system.name},
    )

    docs = search(question, top_k=3)
    search_duration = (time.time() - start_time) * 1000

    logger.log(
        event_type=EventType.SEARCH,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        output_summary=f"Retrieved {len(docs)} documents",
        metadata={"docs_count": len(docs)},
        duration_ms=search_duration,
    )

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "disclosure": AI_DISCLOSURE,
            "sources": [],
            "audit_event_id": query_event.event_id,
        }

    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])
    prompt = f"""Answer the question based on the following documents.

# Reference Documents
{context}

# Question
{question}

# Answer (concisely, based on the documents)"""

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
    print("=== Governance-Compliant RAG ===")
    print(AI_DISCLOSURE)

    result = compliant_rag_answer(
        question="How do you calculate the F1 score?",
        user_id="user_001",
    )

    print(f"Answer:\n{result['answer']}")
    print(f"\nSource documents:")
    for source in result["sources"]:
        print(f"  - {source['title']} (similarity: {source['similarity']})")
    print(f"\nAudit log event ID: {result['audit_event_id']}")
```

---

## 5. EU AI Act Compliance Status

What this implementation covers:

| EU AI Act Article | Content | Implementation |
|------------------|---------|---------------|
| Article 4 | AI literacy | This tutorial itself |
| Article 12 | Record keeping / audit logs | `audit_logger.py` |
| Article 50 | Chatbot transparency disclosure | Display `AI_DISCLOSURE` |
| Article 9 | Risk management | `risk_assessor.py` |
| Annex IV | Technical documentation | `ai_registry.py` |

---

## 6. The Four Principles of AI Governance

```
① Know
  → Use ai_registry.py to track every AI system in use

② Assess
  → Use risk_assessor.py to quantify risk

③ Monitor
  → Use audit_logger.py + Langfuse for real-time monitoring

④ Respond
  → Report to authorities within 72 hours in case of incidents
     (for high-risk systems)
```

---

## FAQ

**Q: Is our RAG system subject to EU AI Act regulation?**
A: It falls under "Limited Risk (chatbot)". We have a transparency obligation (Article 50) to disclose to users that they are interacting with AI. It does not qualify as high-risk, so strict compliance requirements don't apply.

**Q: Does the EU AI Act apply to non-EU companies?**
A: If you provide services to EU users, it applies regardless of where your company is based. If you serve EU users, compliance is required.

**Q: What happens if classified as high-risk?**
A: Conformity assessment, EU Declaration of Conformity, CE marking, and registration in the EU database become required.

---

## Next Steps

- **[Chapter 9: Summary]** — Your next steps as an AI architect
- **ISO 42001 Certification** — AI Management System international standard, maps directly to EU AI Act requirements
- **Incident response planning** — Design the 72-hour reporting flow for when problems occur
- **Third-party risk management** — Include governance for external AI services like the Gemini API
