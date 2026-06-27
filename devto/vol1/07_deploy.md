---
title: Building a RAG System from Scratch — Cloud Deployment with Render and Supabase
published: false
tags: rag, python, deployment, cloud
series: RAG Implementation Guide for AI Architects
---

In the [previous article](https://dev.to), we built an MCP server that any LLM client can connect to locally. In this article, we'll deploy it to the cloud — making it accessible from anywhere.

```
Before: localhost:8000 → pgvector (Docker)
After:  https://your-app.onrender.com/mcp → Supabase (pgvector)
```

Both services are free to start with no credit card required.

---

## Architecture

| Service | Role | Free tier |
|---------|------|-----------|
| **Render** | Host the MCP HTTP server | Persistent web service (sleeps after 15 min) |
| **Supabase** | Managed PostgreSQL + pgvector | 500MB, persistent |

---

## Step 1: Supabase Setup

### Create a project

1. Go to [supabase.com](https://supabase.com) and sign up with GitHub
2. Create a new project — set a database password and choose the Tokyo region
3. Wait 2–3 minutes for provisioning

### Enable pgvector and create the table

Open **SQL Editor** in the Supabase dashboard and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    category    TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    embedding   vector(768)
);

CREATE INDEX IF NOT EXISTS docs_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX ON documents (category);
```

### Get the connection string

Click the **Connect** button at the top of the dashboard (not Settings → Database — the UI has changed).

Select the **Connection pooling** tab and copy the **Transaction** mode URI. It looks like:

```
postgresql://postgres.xxxx:password@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres
```

> **Why port 6543?** The standard port 5432 uses IPv6, which Render doesn't support. The Connection Pooler (port 6543) uses IPv4 and is the correct choice for cloud-to-cloud connections.

---

## Step 2: Migrate Local Data to Supabase

Add the Supabase URL to your `.env`:

```
DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres
```

Then run the migration:

```python
# migrate_to_supabase.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

local_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
local_cur = local_conn.cursor()

supa_conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode="require")
supa_cur = supa_conn.cursor()

local_cur.execute("SELECT title, body, category, embedding FROM documents;")
rows = local_cur.fetchall()
print(f"Migrating {len(rows)} documents...")

for row in rows:
    title, body, category, embedding = row
    supa_cur.execute("""
        INSERT INTO documents (title, body, category, embedding)
        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;
    """, (title, body, category, embedding))

supa_conn.commit()

supa_cur.execute("SELECT COUNT(*) FROM documents;")
count = supa_cur.fetchone()[0]
print(f"Done. Documents in Supabase: {count}")

local_conn.close()
supa_conn.close()
```

```bash
python migrate_to_supabase.py
# Migrating 5 documents...
# Done. Documents in Supabase: 5
```

---

## Step 3: Render-Ready MCP Server

Create `mcp_server/server_render.py`. The only differences from `server.py`:
- DB connection reads from `DATABASE_URL` env var
- Port reads from `PORT` env var (Render sets this automatically)
- Transport is `streamable-http` instead of stdio

```python
# mcp_server/server_render.py
import psycopg2
from google import genai
from google.genai import types as genai_types
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP(
    name="pgvector-search",
    instructions="Document search server using pgvector.",
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Use DATABASE_URL if available (Render/Supabase), fall back to individual vars
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
else:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "vectordb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
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


@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search all document categories for a given query."""
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
    """Search within a specific category (ML, Python, or Cloud)."""
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
    """Return all available categories and document counts."""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents GROUP BY category ORDER BY count DESC;
    """)
    return [{"category": r[0], "count": r[1]} for r in cur.fetchall()]


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sets PORT automatically
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )
```

Push to GitHub:

```bash
git add .
git commit -m "feat: add Render deployment server"
git push origin main
```

---

## Step 4: Deploy to Render

1. Go to [render.com](https://render.com) and sign up with GitHub (no credit card)
2. Click **New → Web Service**
3. Connect your repository
4. Set the following:

| Field | Value |
|-------|-------|
| Name | `pgvector-mcp-server` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python mcp_server/server_render.py` |
| Instance Type | Free |

5. Under **Environment**, add:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | `AIza...` |
| `DATABASE_URL` | The Connection Pooler URI from Supabase (port 6543) |

6. Click **Create Web Service**

### Verify the deployment

Once deployed, visit:

```
https://pgvector-mcp-server.onrender.com/mcp
```

You'll see:

```json
{"jsonrpc":"2.0","id":"server-error","error":{"code":-32600,"message":"Not Acceptable: Client must accept text/event-stream"}}
```

This is correct — it means the server is running. The error appears because browsers aren't MCP clients. Your agent will connect without issues.

> **First request is slow:** Render's free tier sleeps after 15 minutes of inactivity. The first request after sleep takes 30–60 seconds to wake up. Use [UptimeRobot](https://uptimerobot.com) (free) to ping every 5 minutes and prevent sleep.

---

## Step 5: Connect Your Agent

Update `13_mcp_http_agent.py` with the Render URL:

```python
# 13_mcp_http_agent.py
MCP_SERVER_URL = "https://pgvector-mcp-server.onrender.com/mcp"

async def run_agent(task: str):
    async with Client(MCP_SERVER_URL) as mcp_client:  # URL instead of file path
        mcp_tools = await mcp_client.list_tools()
        print(f"Loaded {len(mcp_tools)} tools from remote MCP server")
        # ... rest of the agentic loop is identical ...
```

```bash
python 13_mcp_http_agent.py
# Loaded 3 tools from remote MCP server
# [Step 1]
#   → list_categories({})
# [Step 2]
#   → search_by_category({'query': 'evaluation metrics', 'category': 'ML'})
# [Done in 3 steps]
```

The agent is now querying pgvector on Supabase through an MCP server running on Render — entirely in the cloud.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Network is unreachable` (IPv6) | Using port 5432 | Use Connection Pooler URL (port 6543) |
| `SSL connection required` | Missing sslmode | Add `sslmode="require"` |
| `ModuleNotFoundError` | Missing package | Run `pip freeze > requirements.txt` and push |
| Slow first response | Render sleep | Use UptimeRobot to keep alive |
| `Not Found` at `/` | Wrong URL | Add `/mcp` to the URL |

---

## What We've Built

```
Local development:
  Python agent → stdio → mcp_server/server.py → Docker pgvector

Cloud deployment:
  Python agent → HTTPS → Render (server_render.py) → Supabase pgvector
```

The codebase is identical. The infrastructure changed around it.

In the final article, we'll wrap up the series with a summary of all design decisions and point to Vol.2 — where we cover Evals, Observability, Security, MLOps, Fine-tuning, Multi-Agent, and Governance.

---

*Full source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
