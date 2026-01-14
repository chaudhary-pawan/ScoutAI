from db import supabase

def retrieve_chunks(
    table: str,
    query_embedding: list,
    location_id: int,
    top_k: int = 5
):
    supabase.rpc("set_ivfflat_probes", {"value": 10})

    response = supabase.table(table) \
        .select("content, metadata") \
        .eq("location_id", location_id) \
        .order("embedding <-> %s" % query_embedding) \
        .limit(top_k) \
        .execute()

    return response.data
