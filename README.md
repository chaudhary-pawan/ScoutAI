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

```## 🛠 Tech Stack

Python

Supabase (PostgreSQL + pgvector)

SentenceTransformers (all-MiniLM-L6-v2)

Google Gemini (LLM)

dotenv for environment management
```
---

``` 📁 Project Structure
RAG/
│── rag_pipeline.py     # Single-file RAG + CLI chat
│── .env                # Environment variables
│── requirements.txt
│── README.md


The entire pipeline (intent detection, retrieval, generation, CLI) lives in one file for simplicity and clarity.
```
---

```🔐 Environment Variables

Create a .env file in the project root:

SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-service-role-key>
GEMINI_API_KEY=<your-gemini-api-key>


⚠️ Use Supabase service role key, not anon key.
```
---

```▶️ How to Run

Create and activate a virtual environment

Install dependencies

pip install -r requirements.txt


Run the chatbot

python RAG/rag_pipeline.py
```
---

```💬 Example Chat
You: tell me about kedarkantha trek
AI: Kedarkantha Trek is a popular Himalayan trek in Uttarakhand, known for its scenic forests,
snowy trails, and panoramic summit views. It is suitable for beginners and especially famous
as a winter trek.

You: tell me about most fun experience of scoutripper
AI: Based on available experiences, Scoutripper’s most fun offerings usually include offbeat
camping getaways and immersive valley stays that combine adventure, bonfire nights, and
local cultural experiences.

You: hey how are you
AI: I’m doing great and ready to help you plan your next adventure 🙂

You: bye
AI: Bye! 👋 Safe travels.
```
---
