# Scoutripper RAG

A lightweight Retrieval-Augmented Generation (RAG) pipeline for travel Q&A over Scoutripper content. It embeds queries, retrieves relevant chunks from Supabase (pgvector), builds a grounded prompt, and generates answers with OpenAI.

## Architecture

- `router.py`: Classifies a user query into a content route: `locations`, `treks`, `experiences`, or `multiple`.
- `embedder.py`: Encodes text using SentenceTransformers (`all-MiniLM-L6-v2`).
- `retriever.py`: Fetches top-k similar chunks from Supabase embedding tables using vector distance.
- `prompt.py`: Builds an instruction-following prompt from retrieved chunks + the user query.
- `generator.py`: Calls OpenAI Chat Completions to produce the final answer.
- `db.py`: Initializes a Supabase client from environment variables.
- `main.py`: Orchestrates the end-to-end `rag_pipeline()`.

## Prerequisites

- Python 3.9+
- A Supabase project with pgvector enabled
- OpenAI API access

## Installation

It's recommended to use a virtual environment.

```bash
# From the repo root or RAG directory
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1

pip install --upgrade pip
pip install sentence-transformers openai supabase
```

Notes:
- `sentence-transformers` will install `torch`. For CPU-only, you can rely on the default wheel; avoid committing any `venv` files.
- If you prefer pinning versions, add exact versions to your project requirements.

## Environment Variables

Set these before running:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service or anon key (service key recommended for server-side)
- `OPENAI_API_KEY`: Your OpenAI API key

Example (PowerShell):

```powershell
$env:SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
$env:SUPABASE_KEY = "YOUR_SERVICE_ROLE_KEY"
$env:OPENAI_API_KEY = "sk-..."
```

## Database Expectations

Tables (names used by this repo):

- `locations` with at least columns: `id` (int), `slug` (text)
- `locations_embeddings` | `treks_embeddings` | `experience_embeddings` with columns:
  - `location_id` (int) → FK to `locations.id`
  - `content` (text)
  - `metadata` (jsonb)
  - `embedding` (vector) — pgvector column

Stored procedure (optional but used):
- `set_ivfflat_probes(value int)` — used to adjust HNSW/IVFFlat probe count via `supabase.rpc` for retrieval speed/accuracy trade-off.

## Quickstart

Run the pipeline from Python:

```python
from RAG.main import rag_pipeline

answer = rag_pipeline("What are the best cafes in Manali?")
print(answer)
```

By default, `main.py` currently derives `location_id` by looking up the hardcoded slug `"manali"`. Adapt this to your own entity extraction (NER/regex/LLM) to resolve `location_slug` dynamically.

## How It Works

1. Route: `router.route_query()` classifies the query into one of the content routes.
2. Embed: `embedder.embed_query()` produces a query vector using `all-MiniLM-L6-v2`.
3. Retrieve: `retriever.retrieve_chunks()` queries the appropriate embeddings table, filtered by `location_id` and ordered by `<->` (vector distance).
4. Prompt: `prompt.build_prompt()` composes the instruction + selected context chunks + the user question.
5. Generate: `generator.generate_answer()` uses OpenAI Chat Completions to answer grounded in the provided context.

## Configuration & Customization

- Model choice: change the SentenceTransformers model in `embedder.py`.
- Routing: refine the system prompt and model in `router.py` to better match your domain taxonomy.
- Retrieval: tune `top_k` and probes in `retriever.py`.
- Prompting: edit `prompt.py` for tone/formatting/guardrails.
- Generation: adjust model/temperature in `generator.py`.

## Caveats

- The router currently parses model output via `eval` of a JSON string. Consider replacing with `json.loads` for safety and stricter parsing.
- Ensure that large binaries from virtual environments are ignored by `.gitignore` and never committed to the repo.

## License

This project inherits the repository's root license (see `LICENSE`).
