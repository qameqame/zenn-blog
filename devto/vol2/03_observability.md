---
title: Observability — Tracing RAG and Agents with Langfuse v4
published: false
tags: ai, mlops, llm, python
series: Production Operations Guide for AI Architects
---

## Introduction

In [Chapter 2 (Evals)](./02_evals), we measured answer *quality*. Now we add Observability — making behavior *visible*.

```
[Evals]
Measure answer correctness with scores → quality measurement

[Observability]
Which tools were called, how many times, how long did each take → behavior visualization
```

We'll use **Langfuse v4**, an open-source observability tool. It has a free cloud tier and can also be self-hosted.

> **Note: Langfuse v4 (from March 2026) has a significantly changed API.**
> `langfuse_context`, `update_current_observation`, and `update_current_trace` are deprecated.
> This tutorial is compatible with v4.9+.

---

## What Langfuse Can Do

| Feature | Description |
|---------|-------------|
| **Tracing** | Record execution time and I/O for each RAG/Agent step |
| **Cost management** | Visualize API usage and costs |
| **Dashboard** | Monitor overall quality and latency in real time |

---

## Directory Structure

```
pgvector-tutorial/
├── existing files
├── evals/
│   └── ...                # already created
│
└── observability/
    ├── traced_rag.py      # ★ RAG with tracing (add now)
    └── traced_agent.py    # ★ Agent with tracing (add now)
```

---

## Step 1: Langfuse Setup

### 1-1. Install the Library

```bash
pip install langfuse
pip freeze > requirements.txt
```

### 1-2. Create a Langfuse Account

1. Go to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Sign up with GitHub (free, no credit card required)
3. Create a new project
4. Under "Settings" → "API Keys", retrieve:
   - `LANGFUSE_PUBLIC_KEY` (starts with `pk-lf-...`)
   - `LANGFUSE_SECRET_KEY` (starts with `sk-lf-...`)

### 1-3. Add to `.env`

```
# Existing settings
GEMINI_API_KEY=AIza...
DB_HOST=localhost
...

# Langfuse (new)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> **⚠️ Important: Do not call `get_client()` before `load_dotenv()`**
> Langfuse reads environment variables at initialization. Always call `get_client()` *after* `load_dotenv()`.

---

## Step 2: RAG with Tracing — `observability/traced_rag.py`

Simply add the `@observe()` decorator to automatically record traces.

```python
# observability/traced_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langfuse import get_client, observe
import time

# ── Always call load_dotenv() first ──────────────────────────
load_dotenv()
langfuse = get_client()  # Initialize AFTER load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


@observe()
def get_embedding(text: str) -> list[float]:
    """Trace embedding generation"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


@observe()
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Trace Vector DB search"""
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    results = [
        {"title": r[0], "body": r[1], "similarity": round(r[2], 4)}
        for r in rows
    ]

    # v4: add metadata with update_current_span()
    langfuse.update_current_span(
        metadata={
            "retrieved_count": len(results),
            "top_similarity": results[0]["similarity"] if results else 0,
        }
    )
    return results


@observe(name="llm_generate")
def generate_answer(question: str, context: str) -> str:
    """Trace LLM answer generation"""
    prompt = f"""Answer the question based on the following documents.

# Reference Documents
{context}

# Question
{question}

# Answer (concisely, based on the reference documents)"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


@observe(name="rag_pipeline")
def rag_answer(question: str) -> str:
    """
    Trace the entire RAG pipeline.
    The Langfuse dashboard will show:
    - rag_pipeline (overall)
      ├── search_documents (Vector DB search)
      │   └── get_embedding (Embedding generation)
      └── llm_generate (LLM answer generation)
    """
    langfuse.update_current_span(
        metadata={"question": question, "tags": ["rag", "production"]}
    )

    docs = search_documents(question, top_k=3)
    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])
    answer = generate_answer(question, context)
    return answer


if __name__ == "__main__":
    questions = [
        "How do you calculate the F1 score?",
        "How do you optimize AWS costs?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = rag_answer(question)
        print(f"Answer: {answer[:100]}...")
        time.sleep(5)  # Rate limit safety

    langfuse.flush()
    print("\nTraces sent to Langfuse")
    print("Check the dashboard at https://cloud.langfuse.com")
```

```bash
mkdir observability
python observability/traced_rag.py
```

---

## Step 3: Agent with Tracing — `observability/traced_agent.py`

> **⚠️ Two common gotchas:**
> 1. Call `get_client()` after `load_dotenv()`
> 2. Return `candidates` from `agent_step()` (referenced in `run_agent()`)

```python
# observability/traced_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langfuse import get_client, observe
import time

# ── Always call load_dotenv() first ──────────────────────────
load_dotenv()
langfuse = get_client()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


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


@observe(name="tool_search_documents")
def search_documents(query: str, top_k: int = 3) -> list[dict]:
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


@observe(name="tool_list_categories")
def list_categories() -> list[dict]:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Search documents from the Vector DB.",
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
            name="list_categories",
            description="Get the list of categories in the DB.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    return {"error": f"unknown function: {func_name}"}


@observe(name="agent_step")
def agent_step(contents: list, step_num: int) -> tuple:
    """
    Trace a single Agent step.
    Returns: (part, step_type, candidates)
    Note: candidates must be returned for use in run_agent()
    """
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=[tools]),
            )
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 4:
                wait = (attempt + 1) * 10
                print(f"  Retry {attempt+1}... waiting {wait}s")
                time.sleep(wait)
            else:
                raise

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return None, None, None

    part = candidates[0].content.parts[0]

    if part.function_call:
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        langfuse.update_current_span(
            metadata={"step": step_num, "tool": func_name, "args": str(func_args)}
        )
        return part, "tool_call", candidates
    else:
        langfuse.update_current_span(
            metadata={"step": step_num, "type": "final_answer"}
        )
        return part, "final", candidates


@observe(name="agent_pipeline")
def run_agent(task: str, max_steps: int = 5) -> str:
    """Trace the entire Agent pipeline."""
    langfuse.update_current_span(
        metadata={"task": task, "tags": ["agent", "multi-step"]}
    )

    print(f"\nTask: {task}")
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    step_count = 0

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")

        part, step_type, candidates = agent_step(contents, step + 1)

        if part is None:
            break

        if step_type == "tool_call":
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"  → {func_name}({func_args})")

            result = dispatch(func_name, func_args)
            print(f"  → {len(result) if isinstance(result, list) else result} results")

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
            step_count += 1

        elif step_type == "final":
            text_parts = [
                p.text for p in candidates[0].content.parts
                if hasattr(p, 'text') and p.text
            ]
            answer = "\n".join(text_parts) if text_parts else ""

            langfuse.update_current_span(
                metadata={"total_steps": step_count + 1, "answer_length": len(answer)}
            )
            print(f"\n[Done] Completed in {step + 1} steps")
            return answer

    return "Reached maximum step limit."


if __name__ == "__main__":
    result = run_agent(
        "First check the categories, then give me details about ML evaluation metrics."
    )
    print(f"\nFinal Answer:\n{result[:200]}...")

    langfuse.flush()
    print("\nTraces sent to Langfuse")
    print("Check the dashboard at https://cloud.langfuse.com")
```

```bash
python observability/traced_agent.py
```

---

## Step 4: What You Can See in the Dashboard

After running, open [cloud.langfuse.com](https://cloud.langfuse.com) to see:

**Agent Trace (actual display):**

| Name | Latency |
|------|---------|
| agent_pipeline | 4.40s |
| agent_step [1] | 1.34s (tool: list_categories) |
| tool_list_categories | 0.00s |
| agent_step [2] | 0.66s (tool: search_documents) |
| tool_search_documents | 0.42s |
| agent_step [3] | 1.97s (type: final_answer) |

---

## Langfuse v4 Migration Cheatsheet (from v3)

| v3 (old) | v4 (new) |
|----------|----------|
| `from langfuse.decorators import observe, langfuse_context` | `from langfuse import get_client, observe` |
| `from langfuse import Langfuse` → `Langfuse()` | `from langfuse import get_client` → `get_client()` |
| `langfuse_context.update_current_observation(...)` | `langfuse.update_current_span(...)` |
| `langfuse_context.update_current_trace(...)` | `langfuse.update_current_span(...)` |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Authentication error: initialized without public_key` | `get_client()` called before `load_dotenv()` | Call `get_client()` after `load_dotenv()` |
| `cannot import name 'langfuse_context'` | Deprecated in v4 | Use `from langfuse import get_client, observe` |
| `has no attribute 'update_current_observation'` | Deprecated in v4 | Use `langfuse.update_current_span()` |
| `NameError: name 'candidates' is not defined` | `agent_step()` not returning `candidates` | Use `return part, step_type, candidates` |
| Traces not appearing | `flush()` not called | Add `langfuse.flush()` at end of script |
| `429 RESOURCE_EXHAUSTED` | Gemini free tier limit | Re-run the next day |

---

## Next Steps

- **[Chapter 4: Security]** — Prompt injection defense and guardrail design
- **Integrate with Evals** — Attach Eval scores to traces via Langfuse's Scoring API
- **Continuous monitoring** — Set up production alerts
