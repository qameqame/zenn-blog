---
title: Multi-Agent — Implementing the Orchestrator × Worker Pattern
published: false
tags: ai, mlops, llm, python
series: AI Architect's Production Operations Guide
---

## Introduction

Through [Chapter 6 (Fine-tuning)](./06_finetuning), we focused on improving a single AI system. This chapter introduces **multi-agent** design, where multiple Agents collaborate on complex tasks.

Our previous Agent implementations used a "one Agent does everything" approach. For complex tasks, putting all responsibility on a single Agent has limits.

```
[Single Agent (before)]
User → Agent → Tools → Answer
One Agent makes all decisions and executes everything

[Multi-Agent (now)]
User → Orchestrator → Search Agent
                    → Answer Generation Agent
                    → Quality Check Agent
                 → Final Answer
Each role is specialized and they collaborate
```

**Multi-agent is effective when:**
- Tasks require multiple types of expertise
- Parallel processing can improve speed
- You want a separate Agent to handle quality checking

> **2026 Trend:** Anthropic research shows multi-agent architectures improve task completion rates by 90%. The Orchestrator × Worker pattern has become the production standard.

---

## Architecture for This Chapter

```
User's question
    ↓
[Orchestrator]
  Analyzes task and delegates to two workers
    ↓               ↓
[Search Worker]    [Quality Check Worker]
Searches pgvector  Evaluates answer quality
    ↓               ↓
[Orchestrator]
  Integrates results and generates final answer
    ↓
Final Answer
```

---

## Directory Structure

```
pgvector-tutorial/
├── existing files
└── multiagent/
    ├── search_worker.py      # ★ Search specialist worker
    ├── quality_worker.py     # ★ Quality check specialist worker
    ├── orchestrator.py       # ★ Orchestrator
    └── 14_multiagent.py      # ★ Execution script
```

---

## 1. Search Worker — `multiagent/search_worker.py`

A worker focused exclusively on pgvector search. Single responsibility (search only) keeps the prompt simple and improves accuracy.

```python
# multiagent/search_worker.py
"""
Search specialist worker

Role: Search and return documents from pgvector
Responsibility: Search only (no answer generation)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

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

SEARCH_WORKER_SYSTEM = """You are a document search specialist.
Search for documents related to the given question from pgvector and return the results as-is.

Constraints:
- Do not generate answers (just return search results)
- If no results found, return "No relevant documents found"
- Use category filtering when the category is clear
"""


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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search all categories."""
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """Category-filtered search."""
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE category = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, category, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


def list_categories() -> list[dict]:
    """Get category list."""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


SEARCH_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Search all categories for documents related to a query.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="Search query"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="Number of results"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_by_category",
            description="Search documents in a specific category only.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING),
                    "category": types.Schema(type=types.Type.STRING, description="ML / Python / Cloud"),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                },
                required=["query", "category"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="Return the list of categories in the DB.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    return {"error": f"unknown: {func_name}"}


def run_search_worker(question: str) -> dict:
    """
    Main function for the search worker.

    Returns:
        {
            "docs": [list of retrieved documents],
            "steps": [number of steps executed],
            "worker": "search"
        }
    """
    print(f"  [Search Worker] Question: {question[:40]}...")

    contents = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    docs = []
    steps = 0

    for _ in range(5):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SEARCH_WORKER_SYSTEM,
                        tools=[SEARCH_TOOLS],
                    ),
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep((attempt + 1) * 10)
                else:
                    raise

        candidates = response.candidates
        if not candidates or not candidates[0].content.parts:
            break

        part = candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            result = dispatch(func_name, func_args)
            steps += 1
            print(f"  [Search Worker] {func_name} → {len(result) if isinstance(result, list) else result} results")

            if func_name in ("search_documents", "search_by_category") and isinstance(result, list):
                docs.extend(result)

            contents.append(
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)])
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": result},
                        )
                    )]
                )
            )
        else:
            break

    # Deduplicate by title
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc["title"] not in seen:
            seen.add(doc["title"])
            unique_docs.append(doc)

    print(f"  [Search Worker] Done: retrieved {len(unique_docs)} documents")
    return {"docs": unique_docs, "steps": steps, "worker": "search"}
```

---

## 2. Quality Check Worker — `multiagent/quality_worker.py`

A worker focused exclusively on evaluating answer quality.

```python
# multiagent/quality_worker.py
"""
Quality check specialist worker

Role: Evaluate the quality of generated answers
Responsibility: Quality evaluation only (no search or answer generation)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

QUALITY_WORKER_SYSTEM = """You are a quality evaluation specialist for AI answers.
Evaluate the given "question, reference documents, and answer" set.

Evaluation criteria:
1. Faithfulness (0.0–1.0): Is the answer based on the documents?
2. Relevancy (0.0–1.0): Does it correctly answer the question?
3. Completeness (0.0–1.0): Does it include all necessary information?

Return ONLY the following JSON format:
{
  "faithfulness": 0.0–1.0,
  "relevancy": 0.0–1.0,
  "completeness": 0.0–1.0,
  "overall": 0.0–1.0,
  "feedback": "Note any areas for improvement"
}
"""


def run_quality_worker(question: str, docs: list[dict], answer: str) -> dict:
    """
    Main function for the quality check worker.

    Returns:
        {
            "faithfulness": float,
            "relevancy": float,
            "completeness": float,
            "overall": float,
            "feedback": str,
            "worker": "quality"
        }
    """
    print(f"  [Quality Worker] Evaluating answer quality...")

    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])

    prompt = f"""Please evaluate the following.

# Question
{question}

# Reference Documents
{context}

# Answer to Evaluate
{answer}

Return the evaluation result in JSON format."""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=QUALITY_WORKER_SYSTEM,
                ),
            )
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise

    raw = response.text.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            result["worker"] = "quality"
            print(f"  [Quality Worker] Overall: {result.get('overall', 'N/A')}")
            return result
        except json.JSONDecodeError:
            pass

    print(f"  [Quality Worker] Parse failed, using default values")
    return {
        "faithfulness": 0.5,
        "relevancy": 0.5,
        "completeness": 0.5,
        "overall": 0.5,
        "feedback": "Could not retrieve evaluation",
        "worker": "quality",
    }
```

---

## 3. Orchestrator — `multiagent/orchestrator.py`

The command center that coordinates the two workers.

```python
# multiagent/orchestrator.py
"""
Orchestrator

Role: Receive user questions, direct workers, and integrate results
Responsibility: Task decomposition, worker invocation, result integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv
from multiagent.search_worker import run_search_worker
from multiagent.quality_worker import run_quality_worker
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ORCHESTRATOR_SYSTEM = """You are an orchestrator that decomposes tasks and directs multiple workers.

Your role:
1. Receive the user's question
2. Generate an answer based on documents retrieved by the search worker
3. Review quality check results and improve the answer if needed
4. Return the final answer to the user

Constraints:
- Base answers on the reference documents
- Improve the answer if quality score is below 0.7
- Aim for concise and accurate answers
"""


def generate_answer(question: str, docs: list[dict]) -> str:
    time.sleep(15)  # Rate limit safety (5 requests/min free tier)
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
                config=types.GenerateContentConfig(
                    system_instruction=ORCHESTRATOR_SYSTEM,
                ),
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise
    return "Could not generate an answer."


def improve_answer(question: str, docs: list[dict], answer: str, feedback: str) -> str:
    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])

    prompt = f"""Please improve the following answer.

# Question
{question}

# Reference Documents
{context}

# Current Answer
{answer}

# Improvement Feedback
{feedback}

# Improved Answer"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise
    return answer


def run_orchestrator(question: str) -> dict:
    """
    Main orchestrator function.

    Flow:
    1. Request search from search worker
    2. Generate answer from search results
    3. Request evaluation from quality check worker
    4. Improve answer if score is low
    5. Return final answer
    """
    print(f"\n{'='*60}")
    print(f"Orchestrator started")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # ── Step 1: Request search from search worker ─────────────
    print("\n[Step 1] Delegating to search worker...")
    search_result = run_search_worker(question)
    docs = search_result["docs"]

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "quality": {"overall": 0.0},
            "docs_count": 0,
            "improved": False,
        }

    print(f"  → Retrieved {len(docs)} documents")

    # ── Step 2: Generate answer ───────────────────────────────
    print("\n[Step 2] Generating answer...")
    answer = generate_answer(question, docs)
    print(f"  → Answer generated ({len(answer)} chars)")

    # ── Step 3: Request evaluation from quality worker ────────
    print("\n[Step 3] Delegating to quality check worker...")
    quality = run_quality_worker(question, docs, answer)

    improved = False

    # ── Step 4: Improve if quality is low ────────────────────
    if quality.get("overall", 0) < 0.7:
        print(f"\n[Step 4] Quality score {quality.get('overall')} < 0.7 → Improving answer...")
        feedback = quality.get("feedback", "Please improve the answer")
        answer = improve_answer(question, docs, answer, feedback)
        improved = True
        print(f"  → Answer improvement complete")
    else:
        print(f"\n[Step 4] Quality score {quality.get('overall')} ≥ 0.7 → No improvement needed")

    print(f"\n{'='*60}")
    print("Orchestrator complete")
    print(f"{'='*60}")

    return {
        "answer": answer,
        "quality": quality,
        "docs_count": len(docs),
        "improved": improved,
    }
```

---

## 4. Execution Script — `14_multiagent.py`

```python
# 14_multiagent.py
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent.orchestrator import run_orchestrator

def main():
    questions = [
        "Tell me in detail about ML evaluation metrics",
        "What are the methods for reducing AWS costs?",
    ]

    for question in questions:
        result = run_orchestrator(question)

        print(f"\nFinal Answer:")
        print(result["answer"])
        print(f"\nQuality Scores:")
        q = result["quality"]
        print(f"  Faithfulness:  {q.get('faithfulness', 'N/A')}")
        print(f"  Relevancy:     {q.get('relevancy', 'N/A')}")
        print(f"  Completeness:  {q.get('completeness', 'N/A')}")
        print(f"  Overall:       {q.get('overall', 'N/A')}")
        print(f"  Improved:      {'Yes' if result['improved'] else 'No'}")
        print(f"  Docs used:     {result['docs_count']}")
        print("\n" + "="*60)
        time.sleep(30)  # 30s between questions (rate limit safety)


if __name__ == "__main__":
    main()
```

---

## 5. Running It

```bash
mkdir multiagent
touch multiagent/__init__.py
python 14_multiagent.py
```

Sample output:

```
============================================================
Orchestrator started
Question: Tell me in detail about ML evaluation metrics
============================================================

[Step 1] Delegating to search worker...
  [Search Worker] Question: Tell me in detail about ML ev...
  [Search Worker] list_categories → 3 results
  [Search Worker] search_by_category → 2 results
  [Search Worker] Done: retrieved 2 documents

[Step 2] Generating answer...
  → Answer generated (324 chars)

[Step 3] Delegating to quality check worker...
  [Quality Worker] Evaluating answer quality...
  [Quality Worker] Overall: 0.92

[Step 4] Quality score 0.92 ≥ 0.7 → No improvement needed
============================================================
```

---

## 6. Architecture Pattern Comparison

| Pattern | Structure | Best For |
|---------|-----------|---------|
| **Orchestrator × Worker (this chapter)** | 1 command center + multiple specialists | General task decomposition |
| **Sequential Pipeline** | Agent A → Agent B → Agent C | Document processing, contract generation |
| **Parallel Fan-out** | Multiple Agents execute simultaneously | High-speed research, comparison analysis |
| **Hierarchical** | Upper orchestrator → Lower orchestrator → Workers | Large-scale complex tasks |

---

## 7. Production Considerations

**Watch out for cost explosions**
When the orchestrator receives full results from all workers, the context window balloons. Always have workers return structured JSON summaries — never pass the full text.

**Set step limits**
Always configure a `max_steps` limit on each Agent to prevent runaway loops.

**Error handling**
Use `try/except` per worker so that one worker's failure doesn't halt the entire system.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: multiagent` | Missing `__init__.py` | Run `touch multiagent/__init__.py` |
| `NameError: name 'time' is not defined` | Missing `import time` | Add `import time` at top of `14_multiagent.py` |
| `429 RESOURCE_EXHAUSTED` (per-minute) | Gemini free tier: 5 req/min | Add `time.sleep(15)` at start of `generate_answer()` |
| `429 RESOURCE_EXHAUSTED` (daily) | Gemini free tier: 20 req/day | Re-run the next day |
| Quality worker JSON parse failure | LLM returns malformed JSON | Extract JSON block with `re.search` |
| Cost overrun | Sending full text to workers | Pass summaries / structured JSON |

---

## Next Steps

- **[Chapter 8: Governance]** — Risk management and audit log design for multi-agent systems
- **Parallel execution** — Use `asyncio.gather()` to parallelize the search and quality workers for speed
- **MCP × Multi-Agent** — Expose workers as MCP servers for reusability across projects
