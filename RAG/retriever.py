from db import supabase
from embedder import embed_query
from router import route_query
from config import ROUTE_TO_SOURCE_TYPES

def get_relevant_context_hybrid(user_query: str, location_slug: str):
    # 1. Route the query
    route = route_query(user_query)
    source_types = ROUTE_TO_SOURCE_TYPES.get(
        route, ["locations", "treks", "experiences"]
    )

    # 2. Embed the query
    query_embedding = embed_query(user_query)

    # 3. Call metadata-based RPC
    response = supabase.rpc(
        "match_documents_hybrid",
        {
            "query_embedding": query_embedding,
            "source_types": source_types,
            "location_slug": location_slug,
            "match_threshold": 0.30,
            "match_count": 6
        }
    ).execute()

    return response.data or []
