---
title: Evals — Automatically Measuring RAG Answer Quality
published: false
tags: ai, mlops, llm, python
series: Production Operations Guide for AI Architects
---

## Introduction

In the previous RAG implementation, we built a working system — but we could only verify "is this actually correct?" by reading answers manually.

```
[Before] Manual verification
Ask "How do you calculate F1 score?" → check the answer by eye

[Now — Evals]
Prepare test cases and automatically score quality
```

Evals means preparing an "evaluation dataset" (questions and expected answers) and automatically grading the system's responses.

---

## Three Evaluation Dimensions

RAG system evaluation breaks down into three dimensions:

| Dimension | Meaning | What It Measures |
|-----------|---------|-----------------|
| **Faithfulness** | Grounding | Does the answer rely on retrieved documents? (No hallucinations?) |
| **Answer Relevancy** | Relevance | Is the answer appropriate for the question? |
| **Context Recall** | Retrieval recall | Did the system correctly retrieve documents containing the answer? |

---

## Directory Structure

```
pgvector-tutorial/
├── existing files (01–13)
│
├── evals/
│   ├── dataset.py        # ★ Evaluation dataset definition
│   ├── eval_rag.py       # ★ RAG evaluation
│   ├── eval_agent.py     # ★ Agent evaluation
│   └── report.py         # ★ Evaluation report generation
```

---

## 1. Install Libraries

```bash
pip install pandas tabulate
pip freeze > requirements.txt
```

---

## 2. Evaluation Dataset — `evals/dataset.py`

The evaluation dataset consists of sets of "question, expected answer elements, and expected reference documents."

```python
# evals/dataset.py

EVAL_DATASET = [
    {
        "id": "eval_001",
        "question": "How do you calculate the F1 score?",
        "expected_answer_keywords": ["Precision", "Recall", "harmonic mean", "2"],
        "expected_docs": ["Evaluation metrics for machine learning models"],
        "category": "ML",
    },
    {
        "id": "eval_002",
        "question": "How do you evaluate a model with scikit-learn?",
        "expected_answer_keywords": ["cross_val_score", "classification_report", "scikit-learn"],
        "expected_docs": ["Model evaluation with scikit-learn"],
        "category": "ML",
    },
    {
        "id": "eval_003",
        "question": "How can I reduce AWS costs?",
        "expected_answer_keywords": ["EC2", "spot instances", "cost"],
        "expected_docs": ["AWS cost optimization in practice"],
        "category": "Cloud",
    },
    {
        "id": "eval_004",
        "question": "How do you handle missing values in Pandas?",
        "expected_answer_keywords": ["missing values", "DataFrame", "Pandas"],
        "expected_docs": ["Data preprocessing with Pandas"],
        "category": "Python",
    },
    {
        "id": "eval_005",
        "question": "How do you write a Kubernetes manifest file?",
        "expected_answer_keywords": ["YAML", "Pod", "Kubernetes"],
        "expected_docs": ["Kubernetes Pod basics"],
        "category": "Cloud",
    },
]
```

---

## 3. RAG Evaluation — `evals/eval_rag.py`

```python
# evals/eval_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
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


def get_query_embedding(text: str) -> list[float]:
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
    query_embedding = get_query_embedding(query)
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


def rag_answer(question: str) -> tuple[str, list[dict]]:
    """Generate a RAG answer and return the documents used."""
    docs = search(question, top_k=3)
    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])
    prompt = f"""Answer the question based on the following documents.

# Reference Documents
{context}

# Question
{question}

# Answer (concisely, based on the reference documents)"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text, docs
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise


# ══════════════════════════════════════════
# Evaluation functions
# ══════════════════════════════════════════

def eval_context_recall(retrieved_docs: list[dict], expected_docs: list[str]) -> float:
    """
    Context Recall: Were expected documents included in the search results?
    Score = fraction of expected docs actually retrieved
    """
    retrieved_titles = [d["title"] for d in retrieved_docs]
    hit = sum(1 for expected in expected_docs if expected in retrieved_titles)
    return hit / len(expected_docs) if expected_docs else 0.0


def eval_answer_relevancy(answer: str, keywords: list[str]) -> float:
    """
    Answer Relevancy: Did the answer contain the expected keywords?
    Score = fraction of expected keywords found in the answer
    """
    hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hit / len(keywords) if keywords else 0.0


def eval_faithfulness(answer: str, retrieved_docs: list[dict]) -> float:
    """
    Faithfulness: Is the answer grounded in the retrieved documents?
    Uses LLM-as-a-Judge pattern.
    Score = 0.0–1.0 (LLM-scored)
    """
    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in retrieved_docs])
    prompt = f"""Evaluate the following context and answer.

# Context (retrieved documents)
{context}

# Answer
{answer}

Evaluation criteria:
- Is the answer based on the content of the context?
- Does it add information not present in the context? (hallucination)

Return only a score from 0.0 to 1.0. No explanation. Numbers only."""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            score_text = response.text.strip()
            return float(score_text)
        except (ValueError, Exception) as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                return 0.5  # Default value on eval failure


def run_eval():
    """Evaluate RAG against the full evaluation dataset."""
    results = []

    print("Starting RAG evaluation...")
    print("=" * 60)

    for item in EVAL_DATASET:
        print(f"\n[{item['id']}] {item['question']}")

        answer, retrieved_docs = rag_answer(item["question"])
        time.sleep(2)  # Rate limit safety

        context_recall   = eval_context_recall(retrieved_docs, item["expected_docs"])
        answer_relevancy = eval_answer_relevancy(answer, item["expected_answer_keywords"])
        faithfulness     = eval_faithfulness(answer, retrieved_docs)
        time.sleep(2)

        overall = (context_recall + answer_relevancy + faithfulness) / 3

        result = {
            "id":               item["id"],
            "question":         item["question"][:30] + "...",
            "context_recall":   round(context_recall, 2),
            "answer_relevancy": round(answer_relevancy, 2),
            "faithfulness":     round(faithfulness, 2),
            "overall":          round(overall, 2),
        }
        results.append(result)

        print(f"  Context Recall:   {context_recall:.2f}")
        print(f"  Answer Relevancy: {answer_relevancy:.2f}")
        print(f"  Faithfulness:     {faithfulness:.2f}")
        print(f"  Overall:          {overall:.2f}")

    return results


if __name__ == "__main__":
    results = run_eval()

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    avg_recall    = sum(r["context_recall"]   for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_faith     = sum(r["faithfulness"]     for r in results) / len(results)
    avg_overall   = sum(r["overall"]          for r in results) / len(results)

    print(f"Context Recall:   {avg_recall:.2f}")
    print(f"Answer Relevancy: {avg_relevancy:.2f}")
    print(f"Faithfulness:     {avg_faith:.2f}")
    print(f"Overall:          {avg_overall:.2f}")
```

```bash
python evals/eval_rag.py
```

Sample output:

```
Starting RAG evaluation...
============================================================

[eval_001] How do you calculate the F1 score?
  Context Recall:   1.00
  Answer Relevancy: 1.00
  Faithfulness:     0.92
  Overall:          0.97

[eval_002] How do you evaluate a model with scikit-learn?
  Context Recall:   1.00
  Answer Relevancy: 0.75
  Faithfulness:     0.88
  Overall:          0.88

============================================================
Evaluation Summary
============================================================
Context Recall:   0.95
Answer Relevancy: 0.85
Faithfulness:     0.90
Overall:          0.90
```

---

## 4. Reading the Results

| Score | Meaning |
|-------|---------|
| 0.9+ | Excellent. Production-ready. |
| 0.7–0.9 | Good. Room for improvement. |
| 0.5–0.7 | Needs improvement. Review documents and search config. |
| Under 0.5 | Problem. Reconsider the design. |

### What to do when each metric is low

**When Context Recall is low**
→ Retrieval isn't finding the expected documents
→ Increase `top_k`, revisit document chunking, add metadata filters

**When Answer Relevancy is low**
→ The answer is drifting from the question
→ Improve the prompt, add a system prompt

**When Faithfulness is low**
→ The answer includes information not in the retrieved documents (hallucination)
→ Explicitly state in the prompt: "Do not answer questions not covered in the documents"

---

## 5. The LLM-as-a-Judge Pattern

Using an LLM to score itself, as in `eval_faithfulness()`, is called **LLM-as-a-Judge**.

```
Traditional evaluation:
  Human defines correct answers → rule-based scoring
  → Fast, stable → struggles with nuanced judgment

LLM-as-a-Judge:
  LLM understands evaluation criteria and scores
  → Can handle complex judgments → costs more, scores can vary
```

This implementation combines both:

| Metric | Approach | Reason |
|--------|----------|--------|
| Context Recall | Rule-based (title match) | Clear ground truth |
| Answer Relevancy | Rule-based (keyword match) | Clear ground truth |
| Faithfulness | LLM-as-a-Judge | Hallucination detection requires nuanced judgment |

---

## 6. Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: could not convert string to float` | LLM returned non-numeric output | Strengthen prompt, handle with default value |
| `429 RESOURCE_EXHAUSTED` | Rate limit hit | Increase `time.sleep()` wait time |
| Score always 0 | Keyword variation/mismatch | Revise `expected_answer_keywords` |

---

## Next Steps

- **[Chapter 3: Observability]** — Trace each RAG step with Langfuse and visualize behavior
- **Integrate RAGAS** — `pip install ragas` for a more advanced evaluation framework
- **Continuous evaluation (CI/CD)** — Combine with GitHub Actions in Chapter 5
