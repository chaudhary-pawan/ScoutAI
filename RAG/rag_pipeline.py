import os
import time
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError("Missing environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

# --------------------------------------------------
# INTENT CLASSIFIER
# --------------------------------------------------
def classify_intent(query: str) -> str:
    prompt = f"""
You are an intent classifier.

Classify the user query into ONE category only:

GENERAL → casual talk, greetings, chit-chat
SEARCH → questions that require searching a travel database

Return ONLY one word: GENERAL or SEARCH.

Query:
{query}
"""
    return llm.generate_content(prompt).text.strip().upper()

# --------------------------------------------------
# DOMAIN CLASSIFIER (SOFT)
# --------------------------------------------------
def classify_domain(query: str) -> str:
    prompt = f"""
You are a domain classifier for a travel system.

Domains:
- treks
- experiences
- locations
- multiple

Return ONLY one word.

Query:
{query}
"""
    return llm.generate_content(prompt).text.strip().lower()

# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------
def retrieve_chunks(query: str, source_types: list):
    query_embedding = embedder.encode(query).tolist()

    response = supabase.rpc(
        "match_documents_hybrid",
        {
            "query_embedding": query_embedding,
            "source_types": source_types,
            "match_threshold": 0.30,
            "match_count": 6
        }
    ).execute()

    return response.data or []

# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------
def build_prompt(chunks, user_query, depth="SHORT"):
    context = "\n\n".join(
        f"{c['doc_content']}"
        for c in chunks
    )

    if depth == "SHORT":
        return f"""
You are ScoutAI, a travel assistant.
Answer the user's question clearly and positively in plain text.
Do NOT use bullet points, markdown, or headings.

IMPORTANT RULES:
- Do NOT start the answer with phrases like:
"I cannot find", "I don't know", "No information available".
- If the exact answer is not present, infer the best possible answer
from the available context and phrase it as a suggestion or insight.
- Be honest but optimistic and helpful.

Answer the user's question briefly and clearly in plain text.
Do NOT use bullet points, asterisks, markdown, or headings.
Do NOT include itinerary, packing list, or long explanations
unless the user explicitly asks for details.

Use the context only if needed.

Context:
{context}

Question:
{user_query}

Answer in 5–7 sentences maximum.
"""

    else:  # DETAILED
        return f"""
You are ScoutAI, a travel assistant.

The user wants a detailed explanation.
You may structure the answer with clear sections.
Use headings, but avoid excessive bullet points.

Include itinerary, best time, difficulty, and tips
ONLY if relevant.

Context:
{context}

Question:
{user_query}

Provide a detailed but concise explanation.
"""
# --------------------------------------------------
# Depth Helper Fuction
# --------------------------------------------------
def detect_depth(query: str) -> str:
    deep_keywords = [
        "detailed", "detail", "itinerary", "complete",
        "full", "everything", "explain", "in depth"
    ]

    for word in deep_keywords:
        if word in query.lower():
            return "DETAILED"

    return "SHORT"

# --------------------------------------------------
# MAIN RAG PIPELINE
# --------------------------------------------------
def rag_pipeline(user_query: str) -> str:
    # 1️⃣ Intent Gate
    intent = classify_intent(user_query)

    if intent == "GENERAL":
        return llm.generate_content(user_query).text

    # 2️⃣ Domain Routing (Soft)
    domain = classify_domain(user_query)

    domain_map = {
        "treks": ["treks"],
        "experiences": ["experiences"],
        "locations": ["locations"],
        "multiple": ["treks", "experiences", "locations"]
    }

    source_types = domain_map.get(domain, ["treks", "experiences", "locations"])

    # 3️⃣ Primary Retrieval
    chunks = retrieve_chunks(user_query, source_types)

    # 4️⃣ Fallback Retrieval
    if not chunks:
        chunks = retrieve_chunks(
            user_query,
            ["treks", "experiences", "locations"]
        )

    if not chunks:
        return "Sorry, mujhe is query se related koi information nahi mili."

    # 5️⃣ Generation
    depth = detect_depth(user_query)
    prompt = build_prompt(chunks, user_query, depth)

    return llm.generate_content(prompt).text

# --------------------------------------------------
# CLI CHAT LOOP
# --------------------------------------------------
if __name__ == "__main__":
    print("\n🤖 ScoutAI CLI Chat (type 'exit' to quit)\n")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() in ["exit", "quit", "bye", "ok, bye", "thanks, bye", "thanks", "thank you"]:
            print("AI: Bye! 👋 Safe travels.")
            break

        start = time.time()
        answer = rag_pipeline(user_message)
        latency = round(time.time() - start, 2)

        print("\nAI:", answer)
        print(f"(Latency: {latency}s)\n")
