from retriever import get_relevant_context_hybrid
from prompt import build_prompt
from generator import generate_answer
from db import supabase

def rag_pipeline(user_query: str):
    # Temporary hardcoded location
    location = supabase.table("locations") \
        .select("id") \
        .eq("slug", "manali") \
        .single() \
        .execute()

    location_id = location.data["id"]

    # Retrieve
    chunks = get_relevant_context_hybrid(user_query, location_id)

    if not chunks:
        return "Sorry, mujhe is query se related koi information nahi mili."

    # Prompt
    prompt = build_prompt(chunks, user_query)

    # Generate
    return generate_answer(prompt)


if __name__ == "__main__":
    question = "Tell me about trek clean up drives"
    answer = rag_pipeline(question)
    print(answer)
