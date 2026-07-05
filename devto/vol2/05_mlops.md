---
title: MLOps / LLMOps — CI/CD Pipelines for Continuous Quality Assurance
published: false
tags: ai, mlops, llm, python
series: Production Operations Guide for AI Architects
---

## Introduction

Through [Chapter 4 (Security)](https://dev.to/hiroki-kameyama/security-guardrails-and-prompt-injection-defense-for-production-rag-3h9m), we implemented Evals, Observability, and Security as individual components. In this chapter, we integrate them into a system for *continuous* operations.

LLMOps shares DNA with MLOps but faces fundamentally different challenges. Prompts are code, Evals replace unit tests, provider switching is routine, and costs are unpredictable.

```
[Before] Manually executed scripts
python evals/eval_rag.py   ← run by hand
python security/secure_rag.py ← run by hand

[Now — LLMOps] Automated on every GitHub push
  → Evals check quality
  → Security validation
  → Auto-deploy when quality bar is met
```

The 2026 MLOps Maturity Model identifies Level 2 (CI/CD automation) as delivering the highest ROI — and most organizations sit between Level 1 and 2.

---

## How LLMOps Differs from MLOps

LLMOps adds LLM-specific concerns on top of MLOps: prompt versioning and evaluation, hallucination monitoring, RAG retrieval quality measurement, token cost management, and content safety monitoring.

| MLOps | LLMOps |
|-------|--------|
| Version control model files | Version control prompts |
| Test with accuracy / loss | Test with Evals (LLM-as-a-Judge) |
| Deploy models | Deploy prompt configurations |
| Monitor data drift | Monitor answer quality and costs |

---

## Directory Structure

```
pgvector-tutorial/
├── existing files
│
├── llmops/
│   ├── prompt_registry.py    # ★ Prompt version management
│   ├── ci_eval.py            # ★ CI evaluation script
│   └── cost_tracker.py       # ★ API cost tracking
│
└── .github/
    └── workflows/
        └── llmops.yml        # ★ GitHub Actions CI/CD pipeline
```

---

## 1. Prompt Version Management — `llmops/prompt_registry.py`

Prompts are code. They need version control, diff views, approval workflows, and rollback capability.

```python
# llmops/prompt_registry.py
"""
Prompt version management

Changing a prompt can drastically affect RAG answer quality.
Track which version of each prompt is in use.
"""
import hashlib
from dataclasses import dataclass

PROMPTS = {
    "rag_answer": {
        "v1.0.0": {
            "template": """Answer the question based on the following documents.

# Reference Documents
{context}

# Question
{question}

# Answer""",
            "description": "Initial version",
        },
        "v1.1.0": {
            "template": """Answer the question based on the following documents.
If the information is not in the documents, say "This information is not available in the documents."

# Reference Documents
{context}

# Question
{question}

# Answer (concise, based on the documents)""",
            "description": "Anti-hallucination: explicitly instruct to not answer outside documents",
        },
        "v1.2.0": {
            "template": """You are a document search assistant.
Answer questions based solely on the following documents.

Constraints:
- Do not answer information not in the documents
- Do not speculate or fill gaps
- If unknown, say "This information is not available in the documents"

# Reference Documents
{context}

# Question
{question}

# Answer""",
            "description": "Security hardening: explicit role + constraints in system prompt style",
        },
    }
}


def get_prompt(name: str, version: str = "latest") -> str:
    """Retrieve a prompt template."""
    if name not in PROMPTS:
        raise ValueError(f"Prompt '{name}' not found")

    versions = PROMPTS[name]

    if version == "latest":
        version = sorted(versions.keys())[-1]

    if version not in versions:
        raise ValueError(f"Version '{version}' not found")

    return versions[version]["template"]


def list_versions(name: str) -> list[dict]:
    """Return version list for a prompt."""
    if name not in PROMPTS:
        raise ValueError(f"Prompt '{name}' not found")

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
    """Compare the diff between two versions."""
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


if __name__ == "__main__":
    print("=== Prompt Version List ===\n")
    for version_info in list_versions("rag_answer"):
        print(f"  {version_info['version']} [{version_info['hash']}] - {version_info['description']}")

    print("\n=== Diff: v1.0.0 → v1.2.0 ===")
    diff = compare_versions("rag_answer", "v1.0.0", "v1.2.0")
    print(f"  Lines added: {diff['added_lines']}")
    print(f"  Lines removed: {diff['removed_lines']}")
    print(f"  Char delta: {diff['char_diff']:+d}")

    print("\n=== Latest (v1.2.0) Prompt ===")
    print(get_prompt("rag_answer", "latest"))
```

```bash
mkdir llmops
python llmops/prompt_registry.py
```

---

## 2. CI Evaluation Script — `llmops/ci_eval.py`

This script runs automatically on every GitHub push. It fails CI if quality drops below the threshold.

```python
# llmops/ci_eval.py
"""
CI/CD evaluation script

Runs on every push. Returns exit code 1 if quality
falls below thresholds, causing CI to fail.
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

# ── Quality thresholds (CI fails if below these) ─────────────
QUALITY_THRESHOLDS = {
    "context_recall": 0.80,
    "answer_relevancy": 0.70,
    "overall": 0.75,
}

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
    docs = search(question, top_k=3)
    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])
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


def eval_context_recall(retrieved_docs, expected_docs):
    retrieved_titles = [d["title"] for d in retrieved_docs]
    hit = sum(1 for expected in expected_docs if expected in retrieved_titles)
    return hit / len(expected_docs) if expected_docs else 0.0


def eval_answer_relevancy(answer, keywords):
    hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hit / len(keywords) if keywords else 0.0


def run_ci_eval(prompt_version: str = "latest") -> dict:
    print(f"CI evaluation started: prompt version={prompt_version}")
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
    print("CI Evaluation Report")
    print("=" * 60)
    print(f"Prompt version: {report['prompt_version']}")
    print(f"Context Recall:   {report['metrics']['context_recall']:.3f} (threshold: {QUALITY_THRESHOLDS['context_recall']})")
    print(f"Answer Relevancy: {report['metrics']['answer_relevancy']:.3f} (threshold: {QUALITY_THRESHOLDS['answer_relevancy']})")
    print(f"Overall:          {report['metrics']['overall']:.3f} (threshold: {QUALITY_THRESHOLDS['overall']})")

    with open("llmops/eval_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nReport saved to llmops/eval_report.json")

    if report["passed"]:
        print("\n✅ CI Evaluation: PASSED — Ready to deploy")
        sys.exit(0)
    else:
        print("\n❌ CI Evaluation: FAILED — Quality thresholds not met")
        sys.exit(1)
```

---

## 3. GitHub Actions CI/CD Pipeline — `.github/workflows/llmops.yml`

A pipeline that automatically runs evaluations on every push to GitHub.

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
  # ── Job 1: Security validation ───────────────────────────────
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
        run: pip install -r requirements.txt

      - name: Run input validator tests
        run: python security/input_validator.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}

  # ── Job 2: RAG quality gate (Evals) ─────────────────────────
  eval-gate:
    name: RAG Quality Gate
    runs-on: ubuntu-latest
    needs: security-test

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
        run: python llmops/ci_eval.py
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
        if: always()
        with:
          name: eval-report
          path: llmops/eval_report.json

  # ── Job 3: Deploy (only if quality gate passes) ──────────────
  deploy:
    name: Deploy to Render
    runs-on: ubuntu-latest
    needs: eval-gate
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: Deploy to Render
        run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
```

---

## 4. The LLMOps Pipeline in Full

```
Developer pushes code
    ↓
GitHub Actions triggered
    ↓
[Job 1] Security test
  → Test prompt injection detection with input_validator.py
    ↓ Pass
[Job 2] RAG quality evaluation (Evals gate)
  → Measure Context Recall and Answer Relevancy with ci_eval.py
  → Verify quality meets threshold (Overall ≥ 75%)
  → Save eval report as artifact
    ↓ Pass
[Job 3] Production deployment
  → Auto-deploy to Render
  → Langfuse tracing begins
```

---

## 5. Comparing Prompt Versions

```bash
# Evaluate with v1.1.0
PROMPT_VERSION=v1.1.0 python llmops/ci_eval.py

# Evaluate with v1.2.0
PROMPT_VERSION=v1.2.0 python llmops/ci_eval.py

# Compare scores and set the better version as "latest"
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: llmops` | Path not configured | Check `sys.path.append(...)` |
| CI eval FAILED | Score below threshold | Improve prompt or adjust thresholds |
| GitHub Actions timeout | Gemini rate limit | Increase `time.sleep()` |
| Render Deploy Hook not triggering | Secret not configured | Check GitHub Secrets |

---

## Next Steps

- **[Chapter 6: Fine-tuning]** — Specialize a model for your domain using LoRA
- **Multi-agent** — Design systems where multiple Agents collaborate
- **Governance** — EU AI Act compliance, risk management, audit logs
