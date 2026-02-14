from importlib import metadata
import os
import time
import json
from dotenv import load_dotenv
from supabase import create_client
from huggingface_hub import InferenceClient
import google.generativeai as genai
from functools import lru_cache

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, HF_TOKEN]):
    raise RuntimeError("Missing environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

# ==================================================
# HF INFERENCE EMBEDDING CLIENT (REPLACES TORCH MODEL)
# ==================================================
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

hf_client = InferenceClient(
    model=HF_EMBED_MODEL,
    token=HF_TOKEN
)

import numpy as np

def embed_query_safe(query: str, retries: int = 3) -> list[float]:
    for i in range(retries):
        try:
            emb = hf_client.feature_extraction(query)

            # 🔒 FORCE JSON-SAFE TYPE
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()

            return emb

        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5)


# ==================================================
# 1️⃣ UNIFIED QUERY CLASSIFIER (JSON-SAFE, PROMPT UNCHANGED)
# ==================================================
def classify_query(query: str) -> dict:
    prompt = f"""
You are a STRICT query classifier for a travel RAG system.

Your task is to analyze the user query and return a JSON object
with EXACTLY the following keys:

- intent
- domain
- answer_source

--------------------------------
INTENT CLASSIFICATION
--------------------------------
Return:
- GENERAL → greetings, chit-chat, casual talk, or non-travel queries
- SEARCH → questions about treks, experiences, locations, prices, or details
- RECOMMENDATION → requests for suggestions, recommendations, or "best" options

CRITICAL: When the user asks about "your best", "recommend", or "suggest" 
in the context of TRAVEL OFFERINGS (treks, experiences, locations), 
this is ALWAYS a RECOMMENDATION, NOT GENERAL.

Examples of RECOMMENDATION:
- "your best experience" → RECOMMENDATION (asking for Scoutripper's best experience offering)
- "recommend a trek" → RECOMMENDATION
- "suggest a good location" → RECOMMENDATION  
- "what's the best trek" → RECOMMENDATION
- "can you recommend something" → RECOMMENDATION
- "your top experience" → RECOMMENDATION

Examples of GENERAL (NOT travel-related):
- "hello" → GENERAL
- "how are you" → GENERAL
- "what can you do" → GENERAL

--------------------------------
DOMAIN CLASSIFICATION
--------------------------------
Choose ONE:
- treks
- experiences
- locations
- multiple

--------------------------------
ANSWER SOURCE CLASSIFICATION
--------------------------------
Choose ONE:
- METADATA → factual fields (price, sale_price, duration, altitude, total_distance,
suitable_age, include, exclude, address, map_location,
min_people, max_people, min_day_before_booking, overview
)
- DOC_CONTENT → narrative info (Description, Content, itinerary, overview, FAQs)
- BOTH → narrative + facts

--------------------------------
USER QUERY:
{query}

Return JSON ONLY.
"""

    response = llm.generate_content(prompt).text.strip()

    try:
        return json.loads(response)

    except json.JSONDecodeError:
        # ✅ ONLY FIX: safe fallback (no prompt change)
        return {
            "intent": "SEARCH",
            "domain": "treks",
            "answer_source": "BOTH"
        }


from functools import lru_cache

@lru_cache(maxsize=1024)
def classify_query_cached(query: str) -> dict:
    return classify_query(query)


# ==================================================
# 4️⃣ VECTOR SEARCH
# ==================================================
def retrieve_chunks(query: str, source_types: list):
    query_embedding = embed_query_safe(query)

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
# CORE TREK DETAILS (AUTO-APPENDED)
# ==================================================
def build_core_trek_details(metadata: dict) -> str:
    lines = []

    if metadata.get("address"):
        lines.append(f"📍 Location: {metadata['address']}")

    if metadata.get("altitude"):
        lines.append(f"⛰️ Maximum Altitude: {metadata['altitude']} ft")

    if metadata.get("total_distance"):
        lines.append(f"🥾 Total Trek Distance: {metadata['total_distance']} km")

    if metadata.get("duration"):
        days = max(1, round(int(metadata["duration"]) / 24))
        lines.append(f"🗓️ Ideal Duration: {days} days")

    if not lines:
        return ""

    return "\n".join(lines)



# ==================================================
# PRICE TABLE FORMATTER  ✅ ADD HERE
# ==================================================
def build_price_table(metadata: dict) -> str:
    rows = []

    if metadata.get("price"):
        rows.append(("Regular Price", f"₹{int(metadata['price']):,}"))

    if metadata.get("sale_price"):
        rows.append(("Sale Price", f"₹{int(metadata['sale_price']):,}"))

    if not rows:
        return ""

    lines = ["Price Details", "-" * 28]
    for label, value in rows:
        lines.append(f"{label:<16} {value}")

    return "\n".join(lines)


# ==================================================
# RECOMMENDATION RESPONSE BUILDER
# ==================================================
def build_recommendation_response(metadata: dict, domain: str) -> str:
    """Build a friendly recommendation response"""
    lines = []
    
    title = metadata.get("title", "")
    if not title:
        return "I'd love to recommend something, but I'm having trouble finding the details right now."
    
    # Friendly intro based on domain
    if domain == "treks":
        lines.append(f"I'd recommend the **{title}**!")
    elif domain == "experiences":
        lines.append(f"You should check out **{title}**!")
    elif domain == "locations":
        lines.append(f"**{title}** is a great choice!")
    else:
        lines.append(f"I'd suggest **{title}**!")
    
    # Add overview if available
    overview = metadata.get("overview", {})
    if isinstance(overview, dict) and overview.get("desc"):
        desc_list = overview["desc"]
        if isinstance(desc_list, list) and len(desc_list) > 0:
            # Get first description and limit to 200 chars
            first_desc = desc_list[0]
            if len(first_desc) > 200:
                lines.append(f"\n{first_desc[:200]}...")
            else:
                lines.append(f"\n{first_desc}")
    
    # Add key details
    if metadata.get("address"):
        lines.append(f"\n📍 **Location**: {metadata['address']}")
    
    if metadata.get("price") and metadata.get("sale_price"):
        lines.append(f"💰 **Price**: ₹{int(metadata['sale_price']):,} (Regular: ₹{int(metadata['price']):,})")
    elif metadata.get("sale_price"):
        lines.append(f"💰 **Price**: ₹{int(metadata['sale_price']):,}")
    
    if metadata.get("duration"):
        days = max(1, round(int(metadata["duration"]) / 24))
        lines.append(f"⏱️ **Duration**: {days} days")
    
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
Do not repeat things if the data is shown in core trek details.

Context:
{context}

Question:
{user_query}
"""





# ==================================================
# SESSION HELPERS
# ==================================================
SESSION = {
    "last_entity_title": None,
    "last_entity_slug": None,
    "last_domain": None,
    "core_details_shown": False,
    "last_classification": None
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

# For Streaming responses (if needed)
def stream_gemini(prompt: str):
    response = llm.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text



# ==================================================
# 8️⃣ MAIN RAG PIPELINE
# ==================================================
def rag_pipeline(user_query: str) -> str:
    # 🔁 Follow-up resolution
    if is_followup_query(user_query) and SESSION["last_entity_title"]:
        user_query = f"{user_query} of {SESSION['last_entity_title']}"

    # 🔒 FAST FACTUAL KEYWORD CHECK (DETERMINISTIC)
    FACTUAL_KEYWORDS = {
    "altitude": ["altitude", "height"],
    "sale_price": ["sale price", "discounted price"],
    "price": ["price", "cost","Expense"],
    "total_distance": ["distance", "km"],
    "address": ["location", "where"],
    }

    lower_q = user_query.lower()
    forced_fields = [
        field
        for field, keys in FACTUAL_KEYWORDS.items()
        if any(k in lower_q for k in keys)
        ]

    # ✅ Unified classification (REPLACES intent + domain + answer_source classifiers)
    # ✅ Classification cache (follow-up optimization)
    if is_followup_query(user_query) and SESSION.get("last_classification"):
        classification = SESSION["last_classification"]
    else:
        classification = classify_query_cached(user_query)
        SESSION["last_classification"] = classification
    intent = classification["intent"]
    domain = classification["domain"]
    answer_source = classification["answer_source"]

    
    # Intent gate
    if intent == "GENERAL":
        return llm.generate_content(user_query).text

    # Recommendation intent
    if intent == "RECOMMENDATION":
        chunks = retrieve_chunks(user_query, source_types)
        
        if not chunks:
            # Fallback: try broader search
            chunks = retrieve_chunks(f"popular {domain}", source_types)
        
        if chunks:
            metadata = chunks[0].get("metadata", {}) or {}
            SESSION["last_metadata"] = metadata
            SESSION["last_entity_title"] = metadata.get("title")
            SESSION["last_domain"] = domain
            
            return build_recommendation_response(metadata, domain)
        else:
            domain_singular = domain[:-1] if domain.endswith('s') else domain
            return f"I'd love to recommend a {domain_singular}, but I don't have any information available right now. Could you ask about something specific?"

    # Domain routing (UNCHANGED)
    domain_map = {
        "treks": ["treks"],
        "experiences": ["experiences"],
        "locations": ["locations"],
        "multiple": ["treks", "experiences", "locations"]
    }
    source_types = domain_map.get(domain, ["treks", "experiences", "locations"])

    # ------------------------------
    # 🔁 FOLLOW-UP SHORT-CIRCUIT
    # ------------------------------
    chunks = []

    if is_followup_query(user_query) and SESSION.get("last_metadata"):
    # Reuse previous entity — skip vector search
        metadata = SESSION["last_metadata"]
    else:
        chunks = retrieve_chunks(user_query, source_types)
        if not chunks:
            chunks = retrieve_chunks(user_query, ["treks", "experiences", "locations"])

    if chunks:
        top = chunks[0]
        metadata = top.get("metadata", {}) or {}
        SESSION["last_metadata"] = metadata
        
        # Check if the best match has a very low similarity (weak match)
        similarity = top.get("similarity", 0)
        if similarity < 0.35 and intent == "SEARCH":
            return "I apologize, but I don't have information about that in my database. I can help you with details about treks, experiences, and locations available through Scoutripper. Could you please ask about something else or rephrase your question?"
    else:
        metadata = SESSION.get("last_metadata", {}) or {}
        SESSION["last_metadata"] = metadata

    # ------------------------------
    # 🚫 NO INFORMATION CHECK
    # ------------------------------
    # If no chunks found AND no cached metadata AND it's a SEARCH query
    # Return a clear "no information" message
    if not chunks and not metadata and intent == "SEARCH":
        return "I apologize, but I don't have information about that in my database. I can help you with details about treks, experiences, and locations available through Scoutripper. Could you please ask about something else or rephrase your question?"



    
# ------------------------------
# 🔒 METADATA-PRIORITY DETECTION
# ------------------------------
    if forced_fields:
        requested_fields = forced_fields
    else:
        requested_fields = detect_metadata_fields(user_query)

    metadata_only_query = (
        bool(forced_fields) or
        (bool(requested_fields) and "overview" not in requested_fields)
)



    # ------------------------------
    # CORE TREK DETAILS LOGIC (UNCHANGED)
    # ------------------------------
    core_trek_details = ""

    explicit_core_request = any(
        k in user_query.lower()
        for k in ["overview", "details", "about", "information"]
    )

    if domain == "treks" and (not SESSION["core_details_shown"] or explicit_core_request):
        core_trek_details = build_core_trek_details(metadata)

    # 🔁 Reset session ONLY if a different entity is detected
    new_title = metadata.get("title")
    if should_reset_session_on_entity(new_title):
        SESSION["last_entity_title"] = None
        SESSION["last_entity_slug"] = None
        SESSION["last_domain"] = None
        SESSION["core_details_shown"] = False
        SESSION["last_classification"] = None

    # 💾 Update session memory (UNCHANGED)
    if metadata.get("title"):
        SESSION["last_entity_title"] = metadata["title"]

    if metadata.get("slug"):
        SESSION["last_entity_slug"] = metadata["slug"]

    SESSION["last_domain"] = domain
    save_session()

# ------------------------------
# 🔒 HARD RULE: Metadata overrides LLM routing
# ------------------------------
    if metadata_only_query:
        meta_answer = build_metadata_answer(metadata, requested_fields)
        if meta_answer.strip():
            return meta_answer


    # ------------------------------
    # METADATA ONLY
    # ------------------------------
    if answer_source == "METADATA":
        # 1️⃣ Price fast-path
        if any(k in user_query.lower() for k in ["price", "cost", "sale", "discount"]):
            price_table = build_price_table(metadata)
            if price_table:
                return price_table

        # 2️⃣ Fallback metadata answer
        fields = detect_metadata_fields(user_query)
        if fields:
            meta_answer = build_metadata_answer(metadata, fields)
            if meta_answer.strip():
                if core_trek_details:
                    SESSION["core_details_shown"] = True
                    return meta_answer + "\n\n" + core_trek_details
                return meta_answer

    # ------------------------------
    # DOC_CONTENT ONLY
    # ------------------------------
    if answer_source == "DOC_CONTENT":
        if chunks:
            context = "\n\n".join(c["doc_content"] for c in chunks)
        else:
            context = ""


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

        def doc_content_stream():
            for chunk in stream_gemini(prompt):
                yield chunk

            if core_trek_details:
                yield "\n\n" + core_trek_details
            SESSION["core_details_shown"] = True

        return doc_content_stream()

    # ------------------------------
    # BOTH (metadata + content)
    # ------------------------------
    if answer_source == "BOTH":
        parts = []

        # 1️⃣ Narrative first
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

        parts.append("".join(stream_gemini(prompt)).strip())


        # 2️⃣ Core trek details (before price)
        if core_trek_details:
            parts.append(core_trek_details)
            SESSION["core_details_shown"] = True

        # 3️⃣ Price table
        price_table = build_price_table(metadata)
        if price_table:
            parts.append(price_table)

        final_answer = "\n\n".join(parts)
        return final_answer

    # ------------------------------
    # LLM FALLBACK (UNCHANGED)
    # ------------------------------
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

        print("\nAI:", end=" ", flush=True)

        if hasattr(answer, "__iter__") and not isinstance(answer, str):
            for chunk in answer:
                print(chunk, end="", flush=True)
            print()
        else:
            print(answer)
        print(f"(Latency: {latency}s)\n")
