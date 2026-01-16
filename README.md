# 🧭 ScoutAI — Intent-Driven RAG Chatbot (CLI)

ScoutAI is an **intent-aware, retrieval-augmented chatbot** built for travel platforms like **Scoutripper**.  
It intelligently decides **when to search the database** and **when to answer directly**, ensuring helpful, concise, and user-friendly responses.

The system is designed to **avoid negative responses**, **reduce hallucinations**, and **adapt answer depth based on user intent**.

---

## ✨ Key Features

- 🧠 **Intent Classification**
  - Distinguishes between casual/general queries and database-search queries.
  - Example:
    - “Hey, how are you?” → answered directly by LLM
    - “Tell me about Kedarkantha trek” → searched via RAG

- 🗂 **Domain-Aware Retrieval**
  - Routes search queries to relevant domains:
    - `treks`
    - `experiences`
    - `locations`
  - Uses **soft routing** to avoid missing relevant information.

- 🔍 **Vector Search with Supabase**
  - Embeddings stored in:
    - `treks_embeddings`
    - `experience_embeddings`
    - `locations_embeddings`
  - Unified search via a SQL view and RPC function.

- 🧾 **Metadata-Driven Context**
  - Uses stored metadata (title, description, etc.) to enrich responses.
  - No dependency on relational tables like `locations`.

- 🎯 **Adaptive Answer Depth**
  - Short, concise answers by default.
  - Detailed responses only when explicitly requested (e.g., itinerary, full explanation).

- 🙂 **Soft Failure Handling**
  - Never starts responses with “I don’t know” if related information exists.
  - Provides best possible insights based on available data.

- 💬 **Interactive CLI Chat**
  - Continuous chat loop from terminal.
  - Clean, readable output (no excessive bullets or markdown).

---

## 🏗 Architecture Overview

```text
User Query
   ↓
Intent Classifier (GENERAL / SEARCH)
   ↓
┌──────────────────────────┐
│ GENERAL                  │ → LLM answers directly
└──────────────────────────┘
            OR
┌──────────────────────────┐
│ SEARCH                   │
└──────────────────────────┘
   ↓
Domain Classifier (soft routing)
   ↓
Vector Search (Supabase RPC)
   ↓
Fallback Search (if empty)
   ↓
Prompt Builder (depth-aware)
   ↓
LLM Response
   ↓
CLI Output
```
---

## 🛠 Tech Stack

Python

Supabase (PostgreSQL + pgvector)

SentenceTransformers (all-MiniLM-L6-v2)

Google Gemini (LLM)

dotenv for environment management

---

