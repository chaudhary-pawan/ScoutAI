# Scoutripper
  # Retrieval-Augmented Generation (RAG) System using Supabase & MiniLM

This repository contains a **production-grade Retrieval-Augmented Generation (RAG) pipeline** built using **Supabase (pgvector)**, **Sentence Transformers (MiniLM)**, and **LLM-based query routing**.

The system retrieves semantically relevant information from a vector database, augments the user query with grounded context, and generates accurate responses using an LLM — minimizing hallucinations and improving reliability.

---

## 🚀 Features

- 🔍 **Semantic Search** using `all-MiniLM-L6-v2`
- 🧠 **LLM-based Query Router** (cheap & fast)
- 🗂️ **Metadata-based Filtering** (location_id, slug, etc.)
- ⚡ **IVFFLAT indexing** for fast vector similarity search
- 🧩 **Chunk-based retrieval**
- ✍️ **Context-grounded prompt augmentation**
- 🤖 **LLM response generation**
- 🧪 Modular, clean, and scalable architecture

---

## 🧠 High-Level Architecture

User Query
↓
LLM Router (classification)
↓
Metadata Extraction
↓
Query Embedding (MiniLM)
↓
Vector Search (Supabase + pgvector + IVFFLAT)
↓
Top-K Relevant Chunks
↓
Prompt Augmentation
↓
LLM Answer Generation

yaml
Copy code

---

## 🧱 Tech Stack

| Layer | Technology |
|-----|-----------|
| Vector Database | Supabase (PostgreSQL + pgvector) |
| Indexing | IVFFLAT |
| Embeddings | all-MiniLM-L6-v2 |
| Router LLM | GPT-4o-mini / Gemini Flash |
| Generator LLM | GPT-4o-mini / Gemini |
| Backend | Python |
| Hosting | Vercel / Any cloud |

---

## 📂 Project Structure

rag/
├── db.py # Supabase client
├── embedder.py # MiniLM query embeddings
├── router.py # LLM-based query router
├── retriever.py # Vector similarity search
├── prompt.py # Prompt construction
├── generator.py # LLM response generation
├── main.py # End-to-end RAG pipeline
├── requirements.txt
├── .env.example
└── README.md

yaml
Copy code

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/rag-supabase.git
cd rag-supabase

```

2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt

```

3️⃣ Environment Variables
Create a .env file:

env
Copy code
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key

```

4️⃣ Enable pgvector & IVFFLAT (Supabase)
sql
Copy code
CREATE EXTENSION IF NOT EXISTS vector;

CREATE INDEX locations_embeddings_ivfflat
ON locations_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

```

▶️ Usage
python
Copy code
from main import rag_pipeline

response = rag_pipeline(
    "What are the best places to visit in Manali for 2 days?"
)

print(response)
