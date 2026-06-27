---
title: Building a RAG System from Scratch — MCP: Exposing pgvector as a Reusable Tool Server
published: false
tags: mcp, rag, python, llm
series: RAG Implementation Guide for AI Architects
---

In the [previous article](https://dev.to), we built AI Agents that autonomously search our pgvector database. One limitation remained: the tools were hardcoded inside our Python scripts. Only our code could use them.

**MCP (Model Context Protocol)** fixes this. It turns our search functions into a standalone server that any LLM client can connect to — Claude Desktop, Gemini agents, or any future client.

---

## Tool Use vs MCP

```
Tool Use (what we built):
  Python script → hardcoded functions → Gemini API
  Reusable by: this script only

MCP Server (what we're building):
  Any LLM client → MCP protocol → our server → pgvector
  Reusable by: Claude Desktop, any agent, any language
```

The tools themselves don't change. What changes is where they live and how they're accessed.

---

## MCP's Three Primitives

| Primitive | Role | Our implementation |
|-----------|------|--------------------|
| **Tools** | Functions the LLM can call | `search_documents`, `search_by_category`, `list_categories` |
| **Resources** | Data the LLM can read | `db://categories` (category list) |
| **Prompts** | Reusable prompt templates | `search_prompt(topic)` |

---

## Installing FastMCP

```bash
pip install fastmcp
pip freeze > requirements.txt
```

---

## Step 1: MCP Server — `mcp_server/server.py`

```python
# mcp_server/server.py
import psycopg2
from google import genai
from google.genai import types as genai_types
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP(
    name="pgvector-search",
    instructions="Document search server using pgvector. "
                 "Covers machine learning, Python, and cloud topics.",
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


def get_embedding(text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


# ── Tools ─────────────────────────────────────────────────────
# The @mcp.tool decorator replaces FunctionDeclaration(...) entirely.
# Type hints + docstrings generate the schema automatically.

@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    Search all document categories for a given query.
    Use when the category is unknown or the question spans multiple categories.

    Args:
        query: Search query
        top_k: Number of documents to retrieve (default: 3)
    """
    q = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (q, q, top_k))
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in cur.fetchall()
    ]


@mcp.tool
def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """
    Search within a specific category (ML, Python, or Cloud).
    Use when the category is explicitly mentioned in the question.

    Args:
        query: Search query
        category: Category name — ML, Python, or Cloud
        top_k: Number of documents to retrieve (default: 3)
    """
    q = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents WHERE category = %s
        ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (q, category, q, top_k))
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in cur.fetchall()
    ]


@mcp.tool
def list_categories() -> list[dict]:
    """
    Return all available categories and their document counts.
    Use this first to understand what data is available.
    """
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents GROUP BY category ORDER BY count DESC;
    """)
    return [{"category": r[0], "count": r[1]} for r in cur.fetchall()]


# ── Resources ─────────────────────────────────────────────────
# Resources are read-only data the LLM can access directly.

@mcp.resource("db://categories")
def get_categories_resource() -> str:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents GROUP BY category ORDER BY count DESC;
    """)
    lines = [f"- {r[0]}: {r[1]} documents" for r in cur.fetchall()]
    return "Available categories:\n" + "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────
# Reusable prompt templates.

@mcp.prompt
def search_prompt(topic: str) -> str:
    """Generate a structured search prompt for a given topic."""
    return f"""Research the following topic using the available tools:

Topic: {topic}

Steps:
1. Call list_categories to see what data is available
2. If a relevant category exists, use search_by_category
3. Otherwise use search_documents for a broad search
4. Synthesize the results into a clear answer"""


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()  # stdio mode — standard for Claude Desktop
```

```bash
mkdir mcp_server
touch mcp_server/__init__.py
```

---

## Step 2: Test the Server — `mcp_server/client_test.py`

```python
# mcp_server/client_test.py
import asyncio
from fastmcp import Client

async def test_server():
    async with Client("mcp_server/server.py") as client:

        # List available tools
        tools = await client.list_tools()
        print("=== Available tools ===")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        # List resources
        resources = await client.list_resources()
        print("\n=== Available resources ===")
        for r in resources:
            print(f"  - {r.uri}")

        # Call a tool
        print("\n=== list_categories ===")
        result = await client.call_tool("list_categories", {})
        print(result)

        print("\n=== search_documents ===")
        result = await client.call_tool(
            "search_documents",
            {"query": "ML evaluation metrics", "top_k": 2}
        )
        print(result)

        # Read a resource
        print("\n=== db://categories resource ===")
        content = await client.read_resource("db://categories")
        print(content)

if __name__ == "__main__":
    asyncio.run(test_server())
```

```bash
python mcp_server/client_test.py
# === Available tools ===
#   - search_documents: Search all document categories for a given...
#   - search_by_category: Search within a specific category...
#   - list_categories: Return all available categories...
#
# === list_categories ===
# [{'category': 'ML', 'count': 2}, {'category': 'Cloud', 'count': 2}, ...]
```

---

## Step 3: Agent via MCP — `12_mcp_agent.py`

The biggest difference: tool definitions come from the server, not from hardcoded `FunctionDeclaration` objects.

```python
# 12_mcp_agent.py
import asyncio
from google import genai
from google.genai import types
from fastmcp import Client
from dotenv import load_dotenv
import os
import time

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def run_agent(task: str):
    print(f"\nTask: {task}")
    print("=" * 60)

    async with Client("mcp_server/server.py") as mcp_client:

        # Fetch tool definitions from the server automatically
        mcp_tools = await mcp_client.list_tools()

        # Convert MCP tool definitions to Gemini format
        gemini_tools = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            name: types.Schema(
                                type=types.Type.STRING
                                if schema.get("type") == "string"
                                else types.Type.INTEGER
                                if schema.get("type") == "integer"
                                else types.Type.STRING,
                                description=schema.get("description", ""),
                            )
                            for name, schema in
                            (tool.inputSchema.get("properties") or {}).items()
                        },
                        required=tool.inputSchema.get("required", []),
                    ),
                )
                for tool in mcp_tools
            ]
        )

        print(f"Loaded {len(mcp_tools)} tools from MCP server")

        contents = [types.Content(role="user", parts=[types.Part(text=task)])]

        for step in range(8):
            print(f"\n[Step {step + 1}]")

            for attempt in range(5):
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(tools=[gemini_tools]),
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

                # Execute via MCP server instead of calling locally
                result = await mcp_client.call_tool(func_name, func_args)
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
                text_parts = [
                    p.text for p in candidates[0].content.parts
                    if hasattr(p, 'text') and p.text
                ]
                print(f"\n[Done in {step + 1} steps]")
                return "\n".join(text_parts)

    return "Max steps reached."


async def main():
    result = await run_agent(
        "Check the available categories, then explain ML evaluation metrics in detail."
    )
    print(f"\nFinal answer:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python 12_mcp_agent.py
# Loaded 3 tools from MCP server
# [Step 1]
#   → list_categories({})
# [Step 2]
#   → search_by_category({'query': 'evaluation metrics', 'category': 'ML'})
# [Done in 3 steps]
```

---

## Step 4: Connect to Claude Desktop

If you have Claude Desktop installed, add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pgvector-search": {
      "command": "/path/to/your/project/.venv/bin/python",
      "args": ["/path/to/your/project/mcp_server/server.py"],
      "env": {
        "GEMINI_API_KEY": "AIza...",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "vectordb",
        "DB_USER": "postgres",
        "DB_PASSWORD": "password"
      }
    }
  }
}
```

Restart Claude Desktop. Now you can type "search the pgvector DB for ML evaluation metrics" directly in Claude's chat interface.

> **Note:** Use the full path to your `.venv/bin/python`, not just `python`. Claude Desktop doesn't activate virtual environments automatically.

> **Note:** Claude Desktop currently only supports stdio transport, not HTTP. Use `server.py` (not `server_http.py`) in the config.

---

## Tool Use vs MCP: The Key Difference

```python
# Tool Use — tools defined in code
tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(name="search_documents", ...)  # handwritten
])
result = search_documents(query)  # called directly

# MCP — tools fetched from server
mcp_tools = await mcp_client.list_tools()    # fetched dynamically
result = await mcp_client.call_tool(name, args)  # executed on server
```

The tools are identical. The difference is where they live. MCP makes them a shared infrastructure component rather than a per-project implementation.

---

In the final article of this series, we'll deploy the MCP server to Render and the pgvector database to Supabase — making everything accessible from anywhere.

---

*Full source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
