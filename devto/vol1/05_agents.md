---
title: Building a RAG System from Scratch — AI Agents: Memory, Planning, and Multi-Step Reasoning
published: false
tags: rag, ai, python, llm
series: RAG Implementation Guide for AI Architects
---

In the [previous article](https://dev.to), we gave the LLM the ability to call tools autonomously. Now we'll build a proper **AI Agent** — one that remembers past interactions, plans ahead, and evaluates its own output.

---

## What Makes Something an "Agent"?

An LLM with Tool Use is powerful but stateless — each run starts fresh. An Agent adds three capabilities:

```
Tool Use alone:
  question → LLM decides → tools → answer
  (no memory between runs)

Agent:
  question → plan → execute tools → evaluate → answer
  + remembers past interactions
  + adjusts strategy based on results
```

We'll implement three patterns in order of complexity:

| File | Pattern | What it adds |
|------|---------|-------------|
| `09_agent_basic.py` | ReAct | Reasoning before each action |
| `10_agent_memory.py` | Long-term memory | Persistence across sessions |
| `11_agent_planner.py` | Plan-Execute-Evaluate | Upfront planning + self-evaluation |

---

## Pattern 1: ReAct — `09_agent_basic.py`

ReAct (**Re**asoning + **Act**ing) is the foundational agent pattern. Before taking each action, the agent explicitly reasons about what to do next.

```python
# 09_agent_basic.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time

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
    q = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (q, q, top_k))
    return [{"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
            for r in cur.fetchall()]


def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    q = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents WHERE category = %s
        ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (q, category, q, top_k))
    return [{"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
            for r in cur.fetchall()]


def list_categories() -> list[dict]:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents GROUP BY category ORDER BY count DESC;
    """)
    return [{"category": r[0], "count": r[1]} for r in cur.fetchall()]


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Search all categories. Use when the category is unknown.",
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
            description="Search within a specific category (ML, Python, Cloud). "
                        "Use when the category is explicitly mentioned.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING),
                    "category": types.Schema(type=types.Type.STRING),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                },
                required=["query", "category"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="List all available categories. Use this first to understand what data is available.",
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
    return {"error": f"Unknown: {func_name}"}


# ReAct system prompt — instructs the agent to reason before acting
REACT_SYSTEM = """You are a research agent with access to a document database.

For each step:
1. Think about what information you need
2. Choose the right tool to get it
3. Evaluate whether you have enough to answer

Always start by listing categories to understand what's available.
"""

def run_agent(task: str, max_steps: int = 8) -> str:
    print(f"\nTask: {task}")
    print("=" * 60)

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")

        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=REACT_SYSTEM,
                        tools=[tools],
                    ),
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 4:
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
        else:
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            print(f"\n[Done in {step + 1} steps]")
            return "\n".join(text_parts)

    return "Max steps reached."


result = run_agent(
    "Check what categories are available, then give me a comprehensive "
    "explanation of ML evaluation metrics with Python code examples."
)
print(f"\nFinal answer:\n{result}")
```

```bash
python 09_agent_basic.py
# [Step 1]
#   → list_categories({})
#   → 3 results
# [Step 2]
#   → search_by_category({'query': 'evaluation metrics', 'category': 'ML'})
#   → 2 results
# [Step 3]
#   → search_by_category({'query': 'scikit-learn evaluation', 'category': 'ML'})
#   → 2 results
# [Done in 4 steps]
```

---

## Pattern 2: Long-term Memory — `10_agent_memory.py`

The agent now persists what it learns about you across sessions.

```python
# 10_agent_memory.py (key additions)
import json
from pathlib import Path

MEMORY_FILE = "agent_memory.json"

def load_memory() -> dict:
    if Path(MEMORY_FILE).exists():
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"facts": [], "preferences": [], "history": []}

def save_memory(memory: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def update_memory(memory: dict, task: str, result: str):
    """Ask the LLM to extract facts worth remembering."""
    prompt = f"""Extract any facts about the user worth remembering from this interaction.
Return JSON only: {{"facts": ["..."], "preferences": ["..."]}}

Task: {task}
Result summary: {result[:200]}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    try:
        import re
        raw = response.text.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
            memory["facts"].extend(extracted.get("facts", []))
            memory["preferences"].extend(extracted.get("preferences", []))
            # Keep only the last 20 unique entries
            memory["facts"] = list(set(memory["facts"]))[-20:]
            memory["preferences"] = list(set(memory["preferences"]))[-20:]
    except Exception:
        pass

def run_agent_with_memory(task: str) -> str:
    memory = load_memory()

    # Inject memory into system prompt
    memory_context = ""
    if memory["facts"]:
        memory_context += f"\nKnown facts about the user: {', '.join(memory['facts'])}"
    if memory["preferences"]:
        memory_context += f"\nUser preferences: {', '.join(memory['preferences'])}"

    system = f"""You are a research agent with access to a document database.
{memory_context}
Use this context to personalize your responses."""

    # ... run agent loop (same as before) ...

    result = run_agent(task)  # simplified for brevity

    # Extract and save new memories
    update_memory(memory, task, result)
    memory["history"].append({"task": task, "summary": result[:100]})
    save_memory(memory)

    return result
```

After each session, `agent_memory.json` grows with facts and preferences. The next run picks them up automatically.

---

## Pattern 3: Plan-Execute-Evaluate — `11_agent_planner.py`

The most structured pattern: the agent first makes an explicit plan, executes it step by step, then evaluates whether the goal was achieved.

```python
# 11_agent_planner.py

PLANNER_SYSTEM = """You are a planning agent. Given a task, produce a step-by-step plan.
Return JSON only:
{
  "goal": "...",
  "steps": ["step 1", "step 2", ...],
  "success_criteria": "how to know the task is complete"
}"""

EXECUTOR_SYSTEM = """You are an execution agent. Follow the given plan step by step.
Use the available tools to complete each step.
When all steps are done, produce a final answer."""

EVALUATOR_SYSTEM = """You are an evaluation agent. Assess whether the answer meets the success criteria.
Return JSON only:
{
  "passed": true/false,
  "score": 0.0-1.0,
  "feedback": "what's missing or what to improve"
}"""


def plan(task: str) -> dict:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=task,
        config=types.GenerateContentConfig(system_instruction=PLANNER_SYSTEM),
    )
    import json, re
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    return json.loads(match.group()) if match else {"goal": task, "steps": [task], "success_criteria": "answered"}


def execute(plan_data: dict) -> str:
    task_with_plan = f"""Task: {plan_data['goal']}

Plan:
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(plan_data['steps']))}

Execute each step using the available tools, then provide a final answer."""

    return run_agent(task_with_plan)  # the agentic loop from Pattern 1


def evaluate(task: str, answer: str, success_criteria: str) -> dict:
    prompt = f"""Task: {task}
Success criteria: {success_criteria}
Answer to evaluate: {answer}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=EVALUATOR_SYSTEM),
    )
    import json, re
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    return json.loads(match.group()) if match else {"passed": True, "score": 0.5, "feedback": ""}


def run_planner_agent(task: str) -> str:
    print(f"\nTask: {task}")
    print("=" * 60)

    # Phase 1: Plan
    print("\n[Phase 1: Planning]")
    plan_data = plan(task)
    print(f"Goal: {plan_data['goal']}")
    for i, step in enumerate(plan_data["steps"]):
        print(f"  {i+1}. {step}")

    # Phase 2: Execute
    print("\n[Phase 2: Executing]")
    answer = execute(plan_data)

    # Phase 3: Evaluate
    print("\n[Phase 3: Evaluating]")
    eval_result = evaluate(task, answer, plan_data["success_criteria"])
    print(f"Passed: {eval_result['passed']} | Score: {eval_result['score']}")
    if eval_result.get("feedback"):
        print(f"Feedback: {eval_result['feedback']}")

    return answer


result = run_planner_agent(
    "Research ML evaluation metrics and Python tooling, "
    "then write a practical guide with code examples."
)
print(f"\nFinal answer:\n{result}")
```

---

## When to Use Each Pattern

| Pattern | Best for | Overhead |
|---------|----------|----------|
| **ReAct** | General-purpose agents, most use cases | Low |
| **Long-term memory** | Personalized assistants, repeated users | Medium |
| **Plan-Execute-Evaluate** | Complex multi-step tasks, quality-critical outputs | High |

Start with ReAct. Add memory when you need personalization. Add planning when task complexity demands it.

---

## What We've Built

```
06: LLM decides whether to use a tool
07: LLM routes between multiple tools by reading descriptions
08: LLM uses tools across multiple steps (agentic loop)
09: LLM reasons before each action (ReAct)
10: LLM remembers across sessions (memory)
11: LLM plans, executes, and evaluates (structured pipeline)
```

In the next article, we'll take a step further and expose our search functions as an **MCP (Model Context Protocol) server** — making them reusable by any LLM, not just our Python scripts.

---

*Full source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
