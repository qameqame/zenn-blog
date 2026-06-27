---
title: Building a RAG System from Scratch — Wrap-up and What Comes Next
published: false
tags: rag, llm, python, ai
series: RAG Implementation Guide for AI Architects
---

In this final article, we'll recap what we built across the series, consolidate the design decisions, and point to where to go next.

---

## What We Built

Starting from a blank Python project, we built a complete AI system step by step:

```
01_setup_db.py       pgvector table + extension
02_create_index.py   HNSW index (m=16, ef_construction=64)
03_ingest.py         Embed documents → store in pgvector
04_search.py         Cosine similarity search
05_rag.py            Full RAG pipeline

06_tool_basic.py     LLM decides whether to search
07_tool_multi.py     LLM routes between multiple tools
08_tool_agent.py     Multi-step agentic loop

09_agent_basic.py    ReAct pattern
10_agent_memory.py   Persistent memory across sessions
11_agent_planner.py  Plan → Execute → Evaluate

mcp_server/
  server.py          MCP server (stdio, Claude Desktop)
  server_http.py     MCP server (HTTP)
  server_render.py   MCP server (Render deployment)

12_mcp_agent.py      Agent via MCP (local)
13_mcp_http_agent.py Agent via MCP (cloud)
```

---

## Design Decisions at a Glance

### pgvector over a dedicated vector DB

pgvector integrates with existing PostgreSQL, supports SQL + vector in one query, and handles millions of documents comfortably. Start here and migrate only when you have evidence you need to.

### 768 dimensions

`gemini-embedding-001` outputs 3072 dims by default, but pgvector's HNSW index has a 2000-dim hard limit. 768 dims stays well within bounds with negligible quality loss.

### Asymmetric `task_type`

Use `RETRIEVAL_DOCUMENT` when storing, `RETRIEVAL_QUERY` when searching. The Gemini embedding model is trained to map queries *toward* documents, not to the same point. Using the same task type for both degrades retrieval accuracy.

### HNSW over IVFFlat

HNSW requires no training data, delivers consistent recall at scale, and is faster at query time. IVFFlat is only worth considering under tight memory constraints.

### Tool description is routing logic

The LLM selects tools based on their `description` field. Precise, distinguishing descriptions produce correct tool selection. Vague descriptions produce random behavior.

### The conversation history is the agent's memory

Each tool call and result gets appended to `contents`. The LLM reads the full history on every step — this is how multi-step reasoning works.

### MCP makes tools infrastructure

MCP turns hardcoded functions into a standalone server. Claude Desktop, Gemini agents, and any future client can connect to the same server without duplicating tool definitions.

### Render + Supabase for zero-cost cloud deployment

Render's free web service hosts the MCP server. Supabase's free tier hosts pgvector. The Connection Pooler (port 6543) is mandatory — Render doesn't support the IPv6 used by Supabase's standard port 5432.

---

## The Architecture We Ended Up With

```
Local:
  Claude Desktop
      ↓ stdio
  mcp_server/server.py
      ↓ psycopg2
  pgvector (Docker)

Cloud:
  Python agent (13_mcp_http_agent.py)
      ↓ HTTPS
  Render (server_render.py)
      ↓ PostgreSQL + SSL (port 6543)
  Supabase (pgvector)
      ↓
  Gemini Embedding + LLM
```

---

## What This Series Did Not Cover

This series focused on getting a production-ready RAG system off the ground. Several important topics are out of scope here:

**Evaluation (Evals)** — How do you know if your RAG is actually working? You need automated quality measurement: Context Recall, Answer Relevancy, and Faithfulness scoring.

**Observability** — When something goes wrong in production, how do you debug it? Tracing each step with a tool like Langfuse tells you exactly where latency or quality issues originate.

**Security** — How do you handle adversarial inputs? Prompt injection, jailbreaks, and PII leakage are real threats in any public-facing RAG system.

**MLOps / LLMOps** — How do you ship changes safely? Prompt versioning, CI/CD quality gates, and API cost tracking become essential when the system is in production.

**Fine-tuning** — When the base model doesn't behave the way you need, LoRA fine-tuning lets you adapt it to your domain with surprisingly little data and compute.

**Multi-Agent Systems** — When a single agent isn't enough, orchestrator-worker patterns distribute work across specialized agents.

**Governance** — The EU AI Act is now fully in force. Compliance for a chatbot system means AI disclosure notices, audit logging, and a documented risk assessment.

All of these are covered in **Vol.2** of this series.

---

## Vol.2: Production Operations Guide

The second series picks up where this one leaves off — taking a working RAG system and making it production-grade.

**[AI Production Operations Guide](https://zenn.dev/hkame/books/ai-architect-production)** *(Japanese — English Dev.to series coming soon)*

| Chapter | Topic |
|---------|-------|
| 1 | What "production" actually means |
| 2 | Evals — automated quality measurement |
| 3 | Observability with Langfuse v4 |
| 4 | Security — guardrails and prompt injection defense |
| 5 | MLOps / LLMOps — CI/CD pipeline |
| 6 | Fine-tuning with LoRA |
| 7 | Multi-Agent: orchestrator-worker pattern |
| 8 | Governance — EU AI Act compliance |
| 9 | Wrap-up |

---

## Source Code

Everything built in this series is in one repository:

**[github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)**

The README covers setup, directory structure, and the reasoning behind each design decision.

---

## Series Index

1. [Introduction](https://dev.to)
2. [RAG · Embedding · Vector DB Implementation](https://dev.to)
3. [Design Decisions Explained](https://dev.to)
4. [Tool Use — Autonomous Search](https://dev.to)
5. [AI Agents — Memory and Planning](https://dev.to)
6. [MCP — Reusable Tool Server](https://dev.to)
7. [Cloud Deployment — Render × Supabase](https://dev.to)
8. **Wrap-up and Next Steps** (this article)

---

Thanks for following along. If you found this useful, the GitHub repo and Vol.2 are the best places to continue.
