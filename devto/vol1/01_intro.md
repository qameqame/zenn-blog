---
title: Building a RAG System from Scratch with pgvector and Gemini — Introduction
published: false
tags: rag, llm, python, ai
series: RAG Implementation Guide for AI Architects
---

## What This Guide Covers

When you start building LLM-powered applications, one pattern becomes unavoidable: **RAG (Retrieval-Augmented Generation)**.

LLMs only know what they were trained on. Your company's internal documents, the latest spec sheets, project-specific information — none of that exists in the model. To handle data the model doesn't know, you need a system that retrieves relevant knowledge in real time and injects it into the context. That's RAG.

In this guide, we'll implement a RAG system from scratch using pgvector and Gemini, then extend it step by step through Tool Use, AI Agents, MCP, and cloud deployment.

```
Step 1: Embedding · Vector DB · RAG — core implementation
Step 2: AI Architect perspective — design decisions explained
Step 3: Tool Use — LLM autonomously searches the DB
Step 4: AI Agents — combining multiple tools
Step 5: MCP — exposing tools as a server
Step 6: Cloud deployment — Render × Supabase
```

---

## Three Concepts to Understand First

### Embedding

Computers can't measure "semantic similarity" from raw text. Embedding converts text into a list of numbers (a vector), and semantically similar words produce numerically similar patterns.

```
"dog"  → [0.82, 0.75, 0.10, ...]  768 numbers
"cat"  → [0.78, 0.72, 0.12, ...]  ← similar pattern to "dog"
"bank" → [0.08, 0.10, 0.85, ...]  ← completely different
```

Gemini's embedding model handles this conversion.

### Vector DB

A regular DB searches by keyword matching. A vector DB searches by **numeric distance** — meaning it finds semantically related documents even when the exact words don't match.

```sql
-- Regular search (misses if keywords don't match)
SELECT * FROM docs WHERE body LIKE '%F1 score%';

-- Vector search (finds semantically related docs)
SELECT * FROM docs ORDER BY embedding <=> query_vector LIMIT 3;
```

Search for "how to measure model performance" and it finds "F1 score calculation" — even without matching words. We use **pgvector**, a PostgreSQL extension, for this.

### RAG

LLMs are limited to their training data. RAG is a design pattern that retrieves relevant documents and passes them to the LLM as context, enabling the model to answer questions about data it has never seen.

```
[Plain LLM]  question → answers from training data only
[RAG]        question → search Vector DB → pass results to LLM → grounded answer
```

---

## Who This Is For

- Engineers with Python experience who are new to AI application development
- Anyone who wants to understand RAG, Embedding, and vector search through code
- Anyone who wants to learn hands-on from local implementation to cloud deployment

---

## Tools Used (All Free)

| Tool | Purpose | Free Tier |
|------|---------|-----------|
| Google Gemini API | Embedding generation · answer generation | 1,500 requests/day |
| pgvector (PostgreSQL extension) | Vector storage · search | Unlimited (local) |
| Docker | Run pgvector locally | Unlimited |
| Python 3.12 | Implementation language | — |
| Render | Deploy MCP server | Free web service (with sleep) |
| Supabase | Cloud pgvector | 500MB persistent free |

---

## Where This Fits in the AI Architect Roadmap

This guide focuses on the **Applied** and **Design** phases — the first big implementation step after learning the fundamentals (LLM basics, Prompt Engineering, API/SDK usage).

| | Topic | What we implement |
|--|-------|-------------------|
| ✓ | **RAG** | Full RAG pipeline with pgvector and Gemini |
| ✓ | **Embedding** | Text-to-vector conversion with Gemini Embedding API |
| ✓ | **Vector DB** | Cosine similarity search with pgvector |

Let's get started in the next article with environment setup and the first implementation.

---

## Series Index

1. **Introduction** (this article)
2. RAG · Embedding · Vector DB Implementation
3. Reading RAG Design from an AI Architect's Perspective
4. Tool Use — Letting the LLM Search Autonomously
5. AI Agents — Combining Multiple Tools
6. MCP — Exposing pgvector Search as an MCP Server
7. Cloud Deployment — Render × Supabase
8. Wrap-up and Next Steps

---

*Source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
