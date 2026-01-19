import os
import time
import json
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError("Missing environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==================================================
# 1️⃣ INTENT CLASSIFIER
# ==================================================
def classify_intent(query: str) -> str:
    prompt = f"""
Classify the user intent.

GENERAL → greetings, chit-chat, about AI, casual talk  
SEARCH → questions about treks, experiences, locations, prices, details

Return ONLY one word: GENERAL or SEARCH.

Query:
{query}
"""
    return llm.generate_content(prompt).text.strip().upper()

# ==================================================
# 2️⃣ DOMAIN CLASSIFIER (SOFT)
# ==================================================
def classify_domain(query: str) -> str:
    prompt = f"""
Classify the domain of the query.

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

# ==================================================
# 3️⃣ METADATA FIELD DETECTOR
# ==================================================
def detect_metadata_fields(query: str) -> list:
    prompt = f"""
You are an information extractor for a travel database.

Your task:
Identify EXACTLY which metadata fields are REQUIRED to answer the user query.

Available metadata fields:
price, sale_price, duration, altitude, total_distance,
suitable_age, include, exclude, address, map_location,
min_people, max_people, min_day_before_booking, overview

STRICT RULES (VERY IMPORTANT):
1. If the user explicitly asks for ANY factual detail
(price, cost, duration, altitude, distance, inclusions, exclusions, age, group size, booking rule),
you MUST return ONLY those factual fields.
2. You MUST NOT return "overview" if any factual field applies.
3. Return ["overview"] ONLY when the query is purely descriptive
(e.g. "tell me about", "what is this trek", "describe this experience").
4. Return ONLY a valid JSON array.
5. Do NOT add explanations or text.

Examples:
- "what is the price of vasuki trek" → ["price","sale_price"]
- "altitude and duration of vasuki tal trek" → ["altitude","duration"]
- "what is included and excluded" → ["include","exclude"]
- "tell me about vasuki tal trek" → ["overview"]
- "where is it located" → ["address"]

User query:
{query}
"""

    try:
        response = llm.generate_content(prompt).text.strip()
        return json.loads(response)
    except Exception:
        return []


# ==================================================
# 4️⃣ VECTOR SEARCH
# ==================================================
def retrieve_chunks(query: str, source_types: list):
    query_embedding = embedder.encode(query).tolist()

    response = supabase.rpc(
        "match_documents_hybrid_basic",  # ✅ renamed RPC
        {
            "query_embedding": query_embedding,
            "source_types": source_types,
            "match_threshold": 0.30,
            "match_count": 5
        }
    ).execute()

    return response.data or []


# ==================================================
# 5️⃣ METADATA ANSWER BUILDER
# ==================================================
def build_metadata_answer(metadata: dict, fields: list) -> str:
    lines = []
                            
    for field in fields:
        if field == "price" and metadata.get("price"):
            lines.append(f"Price: ₹{metadata['price']}")

        elif field == "sale_price" and metadata.get("sale_price"):
            lines.append(f"Discounted Price: ₹{metadata['sale_price']}")

        elif field == "duration" and metadata.get("duration"):
            days = max(1, round(int(metadata["duration"]) / 24))
            lines.append(f"Duration: {days} days")

        elif field == "altitude" and metadata.get("altitude"):
            lines.append(f"Maximum Altitude: {metadata['altitude']} ")

        elif field == "total_distance" and metadata.get("total_distance"):
            lines.append(f"Total Distance: {metadata['total_distance']} km")

        elif field == "suitable_age" and metadata.get("suitable_age"):
            lines.append(f"Suitable Age: {metadata['suitable_age']}")

        elif field == "address" and metadata.get("address"):
            lines.append(f"Location: {metadata['address']}")

        elif field == "min_people" and metadata.get("min_people"):
            lines.append(f"Minimum Group Size: {metadata['min_people']}")

        elif field == "max_people" and metadata.get("max_people"):
            lines.append(f"Maximum Group Size: {metadata['max_people']}")

        elif field == "min_day_before_booking" and metadata.get("min_day_before_booking"):
            lines.append(
                f"Booking should be done at least {metadata['min_day_before_booking']} days in advance"
            )

        elif field == "map_location":
            lat, lng = metadata.get("map_lat"), metadata.get("map_lng")
            if lat and lng:
                lines.append(f"Map: https://maps.google.com/?q={lat},{lng}")

        elif field == "include" and metadata.get("include"):
            lines.append("Included:")
            for i in metadata["include"]:
                lines.append(f"- {i['title']}")

        elif field == "exclude" and metadata.get("exclude"):
            lines.append("Excluded:")
            for i in metadata["exclude"]:
                lines.append(f"- {i['title']}")

        elif field == "overview":
            if metadata.get("title"):
                lines.append(f"{metadata['title']} is a popular option offered by Scoutripper.")
            if metadata.get("address"):
                lines.append(f"It is located at {metadata['address']}.")

    return "\n".join(lines)


# ==================================================
# PRICE TABLE FORMATTER  ✅ ADD HERE
# ==================================================
def build_price_table(metadata: dict) -> str:
    rows = []

    if metadata.get("price"):
        rows.append(("Regular Price", f"₹{int(metadata['price']):,}"))

    if metadata.get("sale_price"):
        rows.append(("Discounted Price", f"₹{int(metadata['sale_price']):,}"))

    if not rows:
        return ""

    lines = ["Price Details", "-" * 28]
    for label, value in rows:
        lines.append(f"{label:<16} {value}")

    return "\n".join(lines)


# ==================================================
# 6️⃣ DEPTH DETECTOR
# ==================================================
def detect_depth(query: str) -> str:
    deep_words = ["detail", "itinerary", "complete", "full", "explain", "in depth"]
    return "DETAILED" if any(w in query.lower() for w in deep_words) else "SHORT"

# ==================================================
# 7️⃣ PROMPT BUILDER (LLM FALLBACK)
# ==================================================
def build_prompt(context, user_query, depth):
    if depth == "SHORT":
        return f"""
Answer the question briefly and positively.
Do not use bullet points or markdown.

Context:
{context}

Question:
{user_query}
"""
    return f"""
Give a clear and structured explanation.
Include details only if relevant.

Context:
{context}

Question:
{user_query}
"""


# ==================================================
# 3.5️⃣ ANSWER SOURCE CLASSIFIER
# ==================================================
def classify_answer_source(query: str) -> str:
    prompt = f"""
You are deciding where the answer to a user question is stored.

Possible answer sources:
- METADATA → structured facts like price, duration, altitude, age, inclusions
- DOC_CONTENT → narrative information like itinerary, overview, FAQs, explanations
- BOTH → requires both facts and explanation

Return ONLY one word: METADATA, DOC_CONTENT, or BOTH.

Query:
{query}
"""
    return llm.generate_content(prompt).text.strip().upper()



# ==================================================
# SESSION HELPERS
# ==================================================
SESSION = {
    "last_entity_title": None,
    "last_entity_slug": None,
    "last_domain": None
}

def save_session():
    # placeholder for persistence (SQLite / Supabase later)
    pass

def is_itinerary_query(query: str) -> bool:
    itinerary_keywords = [
        "itinerary",
        "day wise",
        "day-wise",
        "schedule",
        "plan",
        "route plan"
    ]
    q = query.lower()
    return any(k in q for k in itinerary_keywords)


def is_followup_query(query: str) -> bool:
    followup_words = ["it", "its", "this", "that", "these", "those"]
    return any(word in query.lower().split() for word in followup_words)

def should_reset_session_on_entity(new_title: str | None) -> bool:
    if not new_title:
        return False
    if not SESSION["last_entity_title"]:
        return False
    return new_title.lower() != SESSION["last_entity_title"].lower()




# ==================================================
# 8️⃣ MAIN RAG PIPELINE
# ==================================================
def rag_pipeline(user_query: str) -> str:
    # 🔁 Follow-up resolution
    if is_followup_query(user_query) and SESSION["last_entity_title"]:
        user_query = f"{user_query} of {SESSION['last_entity_title']}"

    # Intent gate
    if classify_intent(user_query) == "GENERAL":
        return llm.generate_content(user_query).text

    # Domain routing
    domain = classify_domain(user_query)
    domain_map = {
        "treks": ["treks"],
        "experiences": ["experiences"],
        "locations": ["locations"],
        "multiple": ["treks", "experiences", "locations"]
    }
    source_types = domain_map.get(domain, ["treks", "experiences", "locations"])

    # Retrieve
    chunks = retrieve_chunks(user_query, source_types)
    if not chunks:
        chunks = retrieve_chunks(user_query, ["treks", "experiences", "locations"])

    if not chunks:
        return (
            "I couldn’t find exact information for this, "
            "but I can help you explore similar treks or experiences if you’d like."
        )

    top = chunks[0]
    metadata = top.get("metadata", {}) or {}

    # 🔁 Reset session ONLY if a different entity is detected
    new_title = metadata.get("title")
    if should_reset_session_on_entity(new_title):
        SESSION["last_entity_title"] = None
        SESSION["last_entity_slug"] = None
        SESSION["last_domain"] = None

    # 💾 Update session memory
    if metadata.get("title"):
        SESSION["last_entity_title"] = metadata["title"]

    if metadata.get("slug"):
        SESSION["last_entity_slug"] = metadata["slug"]

    SESSION["last_domain"] = domain
    save_session()   # assuming this already exists in your file

    # Decide answer source
    answer_source = classify_answer_source(user_query)

    if answer_source == "METADATA":
    # 1️⃣ If query is about price, show price table
        if any(k in user_query.lower() for k in ["price", "cost", "sale", "discount"]):
            price_table = build_price_table(metadata)
            if price_table:
                return price_table

    # 2️⃣ Fallback to normal metadata answer
    fields = detect_metadata_fields(user_query)
    if fields:
        meta_answer = build_metadata_answer(metadata, fields)
        if meta_answer.strip():
            return meta_answer


    # ------------------------------
    # DOC_CONTENT ONLY
    # ------------------------------
    if answer_source == "DOC_CONTENT":
        context = "\n\n".join(c["doc_content"] for c in chunks)

    if is_itinerary_query(user_query):
        prompt = f"""
    Provide a clear DAY-WISE ITINERARY.
    Use bullet points or numbered days.
    Do NOT add extra explanations.

    Context:
    {context}
    Question:
    {user_query}
    """
    else:
        depth = detect_depth(user_query)
        prompt = build_prompt(context, user_query, depth)

    return llm.generate_content(prompt).text


    # ------------------------------
    # BOTH (metadata + content)
    # ------------------------------
    if answer_source == "BOTH":
        parts = []

    # 1️⃣ Narrative first (already correct)
    context = "\n\n".join(c["doc_content"] for c in chunks)
    depth = detect_depth(user_query)
    prompt = build_prompt(context, user_query, depth)
    parts.append(llm.generate_content(prompt).text.strip())

    # 2️⃣ Price table second (ONLY if pricing exists)
    price_table = build_price_table(metadata)
    if price_table:
        parts.append(price_table)

    return "\n\n".join(parts)





    # LLM fallback
    context = "\n\n".join(c["doc_content"] for c in chunks)
    depth = detect_depth(user_query)
    prompt = build_prompt(context, user_query, depth)
    return llm.generate_content(prompt).text

# ==================================================
# 9️⃣ CLI CHAT LOOP
# ==================================================
if __name__ == "__main__":
    print("\n🤖 ScoutAI CLI Chat (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye", "thanks", "thank you"]:
            print("AI: Bye! 👋 Safe travels.")
            break

        start = time.time()
        answer = rag_pipeline(user_input)
        latency = round(time.time() - start, 2)

        print("\nAI:", answer)
        print(f"(Latency: {latency}s)\n")
