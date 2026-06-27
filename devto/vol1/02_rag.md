---
title: Building a RAG System from Scratch with pgvector and Gemini — Implementation
published: false
tags: rag, llm, python, pgvector
series: RAG Implementation Guide for AI Architects
---

In the [previous article](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-with-pgvector-and-gemini-introduction-c8i), we covered the three core concepts behind RAG. Now let's build it.

By the end of this article you'll have a working RAG pipeline: documents stored as vectors in pgvector, semantic search retrieving the right context, and Gemini generating grounded answers.

---

## Environment Setup

### Prerequisites

- Python 3.12 (pyenv recommended)
- Docker
- Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com)

### Project setup

```bash
mkdir pgvector-tutorial && cd pgvector-tutorial
pyenv local 3.12.0
python -m venv .venv
source .venv/bin/activate

pip install psycopg2-binary google-genai python-dotenv
pip freeze > requirements.txt
```

> Use `google-genai` (new package), not `google-generativeai` (deprecated).

### Start pgvector with Docker

```bash
docker run -d \
  --name pgvector-demo \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### `.env` file

```
GEMINI_API_KEY=AIza...
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password
```

---

## Directory Structure

We'll build these five files in order:

```
pgvector-tutorial/
├── 01_setup_db.py       # Create table + enable pgvector
├── 02_create_index.py   # HNSW index
├── 03_ingest.py         # Embed documents and store
├── 04_search.py         # Vector search
└── 05_rag.py            # Full RAG pipeline
```

---

## Step 1: Database Setup — `01_setup_db.py`

```python
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id         SERIAL PRIMARY KEY,
        title      TEXT NOT NULL,
        body       TEXT NOT NULL,
        category   TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        embedding  vector(768)
    );
""")

conn.commit()
print("Table created.")
```

```bash
python 01_setup_db.py
```

> **Why 768 dimensions?** `gemini-embedding-001` outputs 3072 dimensions by default, but pgvector's HNSW index has a 2000-dimension limit. Setting `output_dimensionality=768` keeps us well within that limit with negligible quality loss.

---

## Step 2: HNSW Index — `02_create_index.py`

```python
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

cur.execute("""
    CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")

conn.commit()
print("Index created.")
```

```bash
python 02_create_index.py
```

HNSW parameter reference:

| Use case | m | ef_construction |
|----------|---|-----------------|
| Dev / testing | 8 | 32 |
| Production (standard) | 16 | 64 |
| High accuracy | 32 | 128 |

---

## Step 3: Ingest Documents — `03_ingest.py`

```python
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
            task_type="RETRIEVAL_DOCUMENT",  # use RETRIEVAL_DOCUMENT for storage
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def insert_document(title: str, body: str, category: str) -> int:
    embedding = get_embedding(f"{title}\n\n{body}")
    cur.execute("""
        INSERT INTO documents (title, body, category, embedding)
        VALUES (%s, %s, %s, %s) RETURNING id;
    """, (title, body, category, embedding))
    doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id

sample_docs = [
    {
        "title": "ML Model Evaluation Metrics",
        "body": "Precision, Recall, and F1 Score are the key metrics for classification. "
                "The confusion matrix is used to calculate each metric.",
        "category": "ML",
    },
    {
        "title": "Model Evaluation with scikit-learn",
        "body": "Use cross_val_score and classification_report in Python's scikit-learn "
                "library to evaluate machine learning models.",
        "category": "ML",
    },
    {
        "title": "Data Preprocessing with Pandas",
        "body": "Handle missing values, type conversion, and outliers. "
                "Covers basic DataFrame operations and data cleaning workflows.",
        "category": "Python",
    },
    {
        "title": "AWS Cost Optimization in Practice",
        "body": "Reduce costs through EC2 instance type selection, Spot Instance usage, "
                "and deletion of unused resources.",
        "category": "Cloud",
    },
    {
        "title": "Kubernetes Pod Basics",
        "body": "A Pod is the smallest deployable unit in Kubernetes. "
                "Learn how to define Pod manifests in YAML.",
        "category": "Cloud",
    },
]

for doc in sample_docs:
    doc_id = insert_document(doc["title"], doc["body"], doc["category"])
    print(f"Stored: id={doc_id} / {doc['title']}")
```

```bash
python 03_ingest.py
```

> **`task_type` matters:** Use `RETRIEVAL_DOCUMENT` when storing and `RETRIEVAL_QUERY` when searching. This asymmetric setup improves retrieval accuracy.

---

## Step 4: Vector Search — `04_search.py`

```python
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

def get_query_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",  # use RETRIEVAL_QUERY for search
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def search(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    cur.execute("""
        SELECT id, title, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]

# Basic search
results = search("how to measure model accuracy", top_k=3)
for r in results:
    print(f"[{r['similarity']:.4f}] {r['title']} ({r['category']})")
```

```bash
python 04_search.py
# [0.7806] ML Model Evaluation Metrics (ML)
# [0.7423] Model Evaluation with scikit-learn (ML)
# [0.6015] Data Preprocessing with Pandas (Python)
```

The query "how to measure model accuracy" finds "ML Model Evaluation Metrics" even without an exact keyword match — that's the power of semantic search.

---

## Step 5: RAG Pipeline — `05_rag.py`

```python
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

def get_query_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def search(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    cur.execute("""
        SELECT id, title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "body": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]

def rag_answer(question: str) -> str:
    # Step 1: retrieve relevant documents
    docs = search(question, top_k=3)
    if not docs:
        return "No relevant documents found."

    # Step 2: build context
    context = "\n\n".join([f"[{d['title']}]\n{d['body']}" for d in docs])

    # Step 3: build prompt and generate answer
    prompt = f"""Answer the question based on the documents below.

# Documents
{context}

# Question
{question}

# Answer (concise, grounded in the documents above)"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

answer = rag_answer("How do you calculate the F1 score?")
print(answer)
```

```bash
python 05_rag.py
```

---

## Verification Checklist

```bash
# pgvector extension is active
docker exec -it pgvector-demo psql -U postgres -d vectordb -c "\dx"

# Check stored documents
docker exec -it pgvector-demo psql -U postgres -d vectordb \
  -c "SELECT id, title, category FROM documents;"

# Confirm embedding dimensions
docker exec -it pgvector-demo psql -U postgres -d vectordb \
  -c "SELECT id, title, vector_dims(embedding) AS dims FROM documents LIMIT 3;"
# dims should be 768
```

---

## What We Built

```
User question
    ↓
get_query_embedding()   — convert question to vector (RETRIEVAL_QUERY)
    ↓
pgvector search         — find top-3 semantically similar documents
    ↓
Build prompt            — inject retrieved docs as context
    ↓
Gemini generate_content — produce a grounded answer
    ↓
Answer
```

In the next article, we'll look at this pipeline through an AI Architect's lens — why we made these design choices and how to think about scaling them.

---

*Full source code: [github.com/qameqame/pgvector-tutorial](https://github.com/qameqame/pgvector-tutorial)*
