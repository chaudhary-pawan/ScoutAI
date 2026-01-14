from router import route_query
from embedder import embed_query
from retriever import retrieve_chunks
from prompt import build_prompt
from generator import generate_answer
from db import supabase

def rag_pipeline(user_query: str):
    # 1. Route query
    route = route_query(user_query)

    # 2. Extract metadata (example: Manali)
    location_slug = "manali"  # can be extracted via NER / regex / LLM

    location = supabase.table("locations") \
        .select("id") \
        .eq("slug", location_slug) \
        .single() \
        .execute()

    location_id = location.data["id"]

    # 3. Embed query
    query_embedding = embed_query(user_query)

    # 4. Decide table
    table_map = {
        "locations": "locations_embeddings",
        "treks": "treks_embeddings",
        "experiences": "experience_embeddings",
        "multiple": "locations_embeddings"
    }

    table = table_map[route]

    # 5. Retrieve chunks
    chunks = retrieve_chunks(
        table=table,
        query_embedding=query_embedding,
        location_id=location_id
    )

    # 6. Build prompt
    prompt = build_prompt(chunks, user_query)

    # 7. Generate answer
    return generate_answer(prompt)
