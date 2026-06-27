---
title: Building a RAG System from Scratch — Design Decisions Explained
published: false
tags: rag, architecture, llm, python
series: RAG Implementation Guide for AI Architects
---

In the [previous article](https://dev.to), we built a working RAG pipeline. Now let's step back and ask *why* we made each design decision — and what alternatives exist when your requirements change.

---

## The Full Picture

Here's what we built:

```
Ingest phase
  Text → gemini-embedding-001 (RETRIEVAL_DOCUMENT, 768 dims)
       → pgvector (HNSW index, cosine similarity)

Query phase
  Question → gemini-embedding-001 (RETRIEVAL_QUERY, 768 dims)
           → pgvector search (top-k)
           → Gemini 2.5 Flash (answer generation)
```

Every element in this diagram was a choice. Let's examine each one.

---

## Decision 1: pgvector over a Dedicated Vector DB

We used pgvector, a PostgreSQL extension, rather than a purpose-built vector database like Pinecone, Weaviate, or Qdrant.

**Why pgvector works here:**

- Integrates with existing PostgreSQL infrastructure — no new service to operate
- SQL and vector search in the same query: filter by `category`, join with other tables, all in one round-trip
- Handles millions of documents comfortably with HNSW indexing

**When to consider a dedicated vector DB:**

| Signal | Consider moving to |
|--------|-------------------|
| > 10M documents | Pinecone, Weaviate |
| Multi-modal search (text + image) | Weaviate, Qdrant |
| Managed cloud with SLA | Pinecone |
| On-premise, full control | Qdrant |

For most enterprise RAG applications at typical document volumes, pgvector is the right starting point. Migrate when you hit actual limits, not anticipated ones.

---

## Decision 2: 768 Dimensions instead of 3072

`gemini-embedding-001` outputs 3072 dimensions by default. We set `output_dimensionality=768`.

**The constraint:** pgvector's HNSW index has a hard limit of 2000 dimensions.

**Why not 2000?** We chose 768 because:
- It's a well-established embedding size used by BERT and many production systems
- Cosine similarity quality degrades only slightly versus the full 3072 dims for typical retrieval tasks
- Smaller vectors mean faster index builds and lower storage cost

**Dimension vs. quality trade-off:**

| Dimensions | Index build | Storage | Retrieval quality |
|-----------|-------------|---------|-------------------|
| 256 | Fastest | Smallest | Noticeably lower |
| 768 | Fast | Small | Near full quality |
| 1536 | Moderate | Moderate | Full quality |
| 3072 | Slow | Largest | Full quality (no HNSW) |

---

## Decision 3: Asymmetric `task_type`

We used different `task_type` values for ingestion and querying:

```python
# Ingestion
config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", ...)

# Query
config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", ...)
```

**Why this matters:** Gemini's embedding model is trained with asymmetric objectives. A document and a query about the same topic are represented differently in embedding space — the model learns to map queries *toward* relevant documents, not to the same point. Using the same task type for both degrades retrieval accuracy.

This is analogous to how you'd phrase a document differently from a search query in natural language: "F1 Score is the harmonic mean of Precision and Recall" (document) vs. "how to calculate F1" (query).

---

## Decision 4: HNSW over IVFFlat

pgvector supports two index types. We chose HNSW.

| | HNSW | IVFFlat |
|--|------|---------|
| Query speed | Fast | Moderate |
| Build time | Moderate | Fast |
| Memory | Higher | Lower |
| Accuracy at scale | Higher | Lower |
| Requires training data | No | Yes (needs `VACUUM` after inserts) |

HNSW is the better default for production. IVFFlat is worth considering only when you have very tight memory constraints and can afford slower queries.

**HNSW parameter guide:**

```sql
WITH (
    m = 16,              -- max connections per node
    ef_construction = 64 -- search width during build
)
```

- `m`: higher = better recall, more memory. Range: 4–64. Default 16 works for most cases.
- `ef_construction`: higher = better index quality, slower build. Range: 16–512. Default 64 is a good production starting point.

---

## Decision 5: Gemini 2.5 Flash for Generation

We used `gemini-2.5-flash` rather than the more capable `gemini-opus` models.

**Reasoning:**

- Flash has sufficient quality for document-grounded Q&A — the retrieval step does the heavy lifting
- Flash is faster and cheaper (or free-tier eligible during development)
- The generation prompt is constrained: "answer based on these documents" limits hallucination regardless of model capability

**When to upgrade the generation model:**

- Complex multi-step reasoning across many documents
- Synthesis tasks requiring cross-document inference
- When evaluation scores (Faithfulness, Relevancy) are consistently below threshold

**When to upgrade the embedding model:**

- Low Context Recall — the right documents aren't being retrieved
- Evaluation reveals semantic mismatch between queries and stored documents

The embedding model matters more for retrieval quality. The generation model matters more for answer quality. Optimize them independently.

---

## The Scaling Path

This architecture scales predictably:

```
Phase 1 (now): pgvector local → works to ~1M docs
Phase 2:       pgvector + Supabase → managed PostgreSQL, easy scaling
Phase 3:       pgvector + read replicas → horizontal query scaling
Phase 4:       Dedicated vector DB → if you genuinely outgrow pgvector
```

Most teams never reach Phase 4. Start at Phase 1, move when you have evidence you need to.

---

## Common Pitfalls

**Chunking strategy matters more than model choice.** If your documents are long (PDFs, reports), how you split them into chunks dramatically affects retrieval quality. A naive split at 512 tokens often breaks context mid-sentence. Consider semantic chunking or overlap.

**Don't embed the question alone.** For complex questions, consider HyDE (Hypothetical Document Embedding): generate a hypothetical answer to the question, embed that, then search. This often retrieves better documents than embedding the raw question.

**Reranking improves precision.** After vector search returns top-k candidates, a cross-encoder reranker (like Cohere Rerank) re-scores them for precision. Add this when recall is good but final answer quality is inconsistent.

---

In the next article, we'll give the LLM the ability to call these search functions autonomously using Tool Use.

---

*Full source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
