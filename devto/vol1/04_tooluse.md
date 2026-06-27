---
title: Building a RAG System from Scratch — Tool Use: Let the LLM Search Autonomously
published: false
tags: rag, llm, python, ai
series: RAG Implementation Guide for AI Architects
---

In the [previous article](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-design-decisions-explained-40hd), we examined the design decisions behind our RAG pipeline. Now we'll give the LLM the ability to call our search functions autonomously — this is **Tool Use**.

---

## What Changes with Tool Use

In our RAG pipeline so far, we always called `search()` before generating an answer. The flow was hardcoded:

```
question → search() → generate_answer()
```

With Tool Use, the LLM decides whether to search, what to search for, and when it has enough information to answer:

```
question → LLM decides → search() if needed → LLM decides → answer
```

This matters when:
- The question might already be answerable without retrieval
- The right search query isn't the same as the user's question
- Multiple searches with different queries improve the answer

---

## How Tool Use Works

The LLM is given a list of available functions with their signatures and descriptions. It responds with either:
1. A `function_call` — "call this function with these arguments"
2. A text answer — "I have enough information to answer directly"

Your code executes the function call and sends the result back. The LLM then decides whether to call another function or produce a final answer.

```
You → LLM: "here are available tools + user question"
LLM → You: function_call { name: "search_documents", args: { query: "F1 score" } }
You → execute search_documents("F1 score") → results
You → LLM: function_result { ... }
LLM → You: "The F1 score is calculated as..."
```

---

## Step 1: Basic Tool Use — `06_tool_basic.py`

```python
# 06_tool_basic.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
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


# ── Tool definition ──────────────────────────────────────────
# Instead of calling search_documents() directly, we describe it to the LLM.
# The description is what the LLM uses to decide when to call it.
tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Search documents in the vector DB for a given query. "
                        "Use this when you need information to answer the question.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of documents to retrieve (default: 3)",
                    ),
                },
                required=["query"],
            ),
        ),
    ]
)


def run(question: str):
    print(f"Question: {question}\n")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(tools=[tools]),
    )

    part = response.candidates[0].content.parts[0]

    if part.function_call:
        # LLM decided to call a tool
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        print(f"→ LLM called: {func_name}({func_args})")

        result = search_documents(**func_args)
        print(f"→ Retrieved {len(result)} documents")
        print(f"→ Top result: {result[0]['title']}")
    else:
        # LLM answered directly without searching
        print(f"→ LLM answered directly (no search needed)")
        print(part.text)


run("How do you calculate the F1 score?")
run("What is 2 + 2?")  # LLM should answer this without searching
```

```bash
python 06_tool_basic.py
# Question: How do you calculate the F1 score?
# → LLM called: search_documents({'query': 'F1 score calculation'})
# → Retrieved 3 documents
# → Top result: ML Model Evaluation Metrics
#
# Question: What is 2 + 2?
# → LLM answered directly (no search needed)
# → 4
```

The LLM correctly decides when to search and when not to.

---

## Step 2: Multiple Tools — `07_tool_multi.py`

Now we give the LLM two tools: one for general search and one for category-filtered search. The LLM picks the right one based on the question.

```python
# 07_tool_multi.py (key additions)

def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
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


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Search all categories when the category is unknown "
                        "or the question spans multiple categories.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_by_category",
            description="Search within a specific category (ML, Python, or Cloud). "
                        "Use this when the question clearly targets one category.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Category name: ML, Python, or Cloud",
                    ),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                },
                required=["query", "category"],
            ),
        ),
    ]
)
```

> **The description is the routing logic.** The LLM reads the `description` field to decide which tool to call. Write descriptions that clearly distinguish when to use each tool — this is prompt engineering for tool selection.

---

## Step 3: Agentic Loop — `08_tool_agent.py`

The real power of Tool Use is the **agentic loop**: the LLM can call multiple tools in sequence, building up context before producing a final answer.

```python
# 08_tool_agent.py

def dispatch(func_name: str, func_args: dict):
    """Route function calls to the right Python function."""
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    return {"error": f"Unknown function: {func_name}"}


def run_agent(task: str, max_steps: int = 8):
    print(f"\nTask: {task}")
    print("=" * 60)

    # Conversation history — this is what enables multi-step reasoning
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[tools]),
        )

        part = response.candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"[Step {step+1}] → {func_name}({func_args})")

            result = dispatch(func_name, func_args)

            # Append the tool call and result to conversation history
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
            # LLM produced a final answer
            text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
            print(f"\n[Done in {step+1} steps]")
            return "\n".join(text_parts)

    return "Max steps reached."


result = run_agent(
    "What evaluation metrics are available for ML models? "
    "Show me both the metric names and how to implement them in Python."
)
print(f"\nFinal answer:\n{result}")
```

```bash
python 08_tool_agent.py
# Task: What evaluation metrics are available for ML models?...
# [Step 1] → search_by_category({'query': 'ML evaluation metrics', 'category': 'ML'})
# [Step 2] → search_by_category({'query': 'scikit-learn model evaluation', 'category': 'ML'})
# [Done in 3 steps]
# Final answer: ML models are evaluated using...
```

The agent searched twice with different queries, gathered complementary information, then synthesized a comprehensive answer.

---

## Key Patterns

**The conversation history is the agent's memory.** Each tool call and its result gets appended to `contents`. The LLM sees the full history on every step, which is how it knows what it has already retrieved and what it still needs.

**`dispatch()` is the bridge.** It maps function names (strings from the LLM) to actual Python functions. Keep it simple and exhaustive — every tool the LLM can call must have an entry here.

**The `description` field does the routing.** Spend time on tool descriptions. A vague description leads to random tool selection. A precise description ("use this when the category is explicitly mentioned") leads to correct routing almost every time.

---

## What We've Added

```
Before Tool Use:
  hardcoded: question → search → answer

After Tool Use:
  autonomous: question → LLM decides → search (maybe) → LLM decides → answer
```

In the next article, we'll build a full **AI Agent** with memory, planning, and multiple tools working together.

---

*Full source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
