---
title: Summary — Your Next Steps as an AI Architect
published: false
tags: ai, mlops, llm, python
series: AI Architect's Production Operations Guide
---

## What We Built in This Guide

In the previous guide, we went from RAG to cloud deployment. In this guide, we systematically implemented everything needed to take that system to *production*.

```
evals/
  dataset.py          # Evaluation dataset
  eval_rag.py         # Context Recall · Relevancy · Faithfulness

observability/
  traced_rag.py       # RAG pipeline tracing with @observe() (Langfuse v4)
  traced_agent.py     # Trace each Agent step

security/
  input_validator.py  # Prompt injection detection
  output_validator.py # PII masking and leakage detection
  guardrails.py       # Rate limiting, security log integration
  secure_rag.py       # RAG with guardrails

llmops/
  prompt_registry.py  # Prompt version management (v1.0–v1.2)
  ci_eval.py          # Quality gate (Overall ≥ 75% to deploy)
  cost_tracker.py     # API cost tracking

finetuning/
  prepare_dataset.py  # Convert to Alpaca format
  train_lora.py       # LoRA fine-tuning (r=8, 2 min on CPU)
  inference.py        # Compare with base model

multiagent/
  search_worker.py    # Search specialist worker
  quality_worker.py   # Quality check specialist worker
  orchestrator.py     # Task decomposition and result integration
  14_multiagent.py    # Execution script

governance/
  ai_registry.py      # AI system inventory
  risk_assessor.py    # Risk assessment (score 0.18 → LOW)
  audit_logger.py     # Audit log (Article 12 compliant)
  compliant_rag.py    # RAG with AI disclosure (Article 50 compliant)
```

---

## Key Design Decisions from Each Chapter

### Chapter 2: Evals

Combining rule-based (Context Recall, Answer Relevancy) with LLM-as-a-Judge (Faithfulness) strikes the right balance between speed, cost, and coverage.

### Chapter 3: Observability (Langfuse v4)

Adding `@observe()` decorators is all it takes to start recording traces. The critical v4 change: you must call `get_client()` *after* `load_dotenv()`.

### Chapter 4: Security

Defense in Depth is the principle: Input validation → System prompt → Output validation → Rate limiting — four layers of protection.

### Chapter 5: MLOps / LLMOps

On every push to GitHub, Evals run automatically. Only when the quality threshold (Overall ≥ 75%) is met does the system auto-deploy to Render.

### Chapter 6: Fine-tuning (LoRA)

Only 0.09% of parameters (2.6M out of 2.7B) are trained. Completes in under 2 minutes on CPU. Even 8 samples show learning trends, but 100+ are needed for practical quality improvement.

### Chapter 7: Multi-Agent

The **single responsibility principle** is key. When the Search Worker, Quality Check Worker, and Orchestrator each focus on exactly one thing, each Agent's prompt stays simple and LLM output quality improves.

### Chapter 8: Governance

Our RAG system falls under EU AI Act "Limited Risk (chatbot)." Risk score: 0.18 (LOW). Implementing AI disclosure (Article 50) and audit logging (Article 12) establishes the compliance foundation.

---

## The Full Picture: Two Guides

```
[Guide 1: RAG Implementation Guide for AI Architects]
Foundation → RAG → Tool Use → Agents → MCP → Deployment
"Building systems that work"

[Guide 2: AI Architect's Production Operations Guide (this guide)]
Evals → Observability → Security → MLOps
→ Fine-tuning → Multi-Agent → Governance
"Making systems that work in production"
```

An AI architect's job isn't just "build something that works." It's designing the systems to *measure quality*, *make behavior visible*, *defend against attacks*, *improve continuously*, *coordinate multiple Agents*, and *comply with regulations*.

---

## Files Implemented in Total

| Phase | File Count | Key Technologies |
|-------|-----------|-----------------|
| Evals | 2 | LLM-as-a-Judge |
| Observability | 2 | Langfuse v4 |
| Security | 4 | Regex, guardrails |
| MLOps | 3 | GitHub Actions, prompt management |
| Fine-tuning | 3 | LoRA, Hugging Face |
| Multi-Agent | 4 | Orchestrator, Workers |
| Governance | 4 | EU AI Act, audit logs |
| **Total** | **22** | |

---

## References

- [Previous guide: RAG Implementation Guide for AI Architects](https://dev.to/hkame)
- [Langfuse Official Documentation (v4)](https://langfuse.com/docs)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [EU AI Act Official Text](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Anthropic Multi-Agent Design Guide](https://docs.anthropic.com/en/docs/build-with-claude/agents)
- [Source code: github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)
