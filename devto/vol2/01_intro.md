---
title: Taking RAG to Production — Evals, Observability, Security, and Beyond (Introduction)
published: false
tags: ai, mlops, llm, python
series: AI Architect's Production Operations Guide
---

## About This Guide

In the previous guide, [*RAG Implementation Guide for AI Architects*](https://dev.to/hkame), we implemented a RAG system from scratch using pgvector and Gemini, then extended it through Tool Use, AI Agents, MCP, and cloud deployment.

This guide is the sequel. It takes you from *"building a system that works"* to *"making a system that works in production"*.

```
[Previous Guide]
RAG → Tool Use → AI Agents → MCP → Render × Supabase deployment

[This Guide]
Evals → Observability → Security → MLOps → Fine-tuning
→ Multi-Agent → Governance
```

---

## Why Production Operations Are Hard

After implementing RAG or Agents, you'll inevitably hit these problems when trying to go live:

**Quality problems**
Manually checking "is this answer correct?" doesn't scale. You need an automated system to measure quality.

**Visibility problems**
When something goes wrong in production, you can't diagnose it if you can't track "what happened at which step."

**Security problems**
When accepting requests from external users, you need to handle prompt injection attacks and prohibited content.

**Continuous improvement problems**
If you have no way to verify "did this actually get better?" after improving a prompt, your iteration cycle stalls.

**Model problems**
Sometimes a general-purpose model isn't enough — you need a model specialized for your specific domain.

---

## Guide Structure

Each chapter can be read independently. The content assumes the previous guide's implementation (pgvector, Gemini, RAG), but you can read for conceptual understanding alone.

| Chapter | Theme | Problem Solved |
|---------|-------|---------------|
| Ch. 2 | Evals | Automated measurement of answer quality |
| Ch. 3 | Observability | Tracing and cost management |
| Ch. 4 | Security | Guardrails and attack defense |
| Ch. 5 | MLOps / LLMOps | CI/CD and prompt management |
| Ch. 6 | Fine-tuning | Domain-specific model specialization |
| Ch. 7 | Multi-Agent | Orchestrator × Worker architecture |
| Ch. 8 | Governance | EU AI Act compliance, audit logs |

---

## Prerequisites

- Completed the previous guide's pgvector tutorial
- Python 3.11, Docker, pgvector environment set up
- `GEMINI_API_KEY` configured in `.env`

---

## Tools Used

| Tool | Purpose | Free Tier |
|------|---------|-----------|
| Google Gemini API | LLM + Embedding | 1,500 requests/day |
| pgvector | Vector DB | Unlimited (local) |
| Langfuse | Observability | Free cloud tier available |
| GitHub Actions | CI/CD pipeline | 2,000 minutes/month (free) |
| Hugging Face | Fine-tuning models | Free |

Let's start with Chapter 2: Evals.
