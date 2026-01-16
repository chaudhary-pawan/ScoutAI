from retriever import get_relevant_context_hybrid
from prompt import build_prompt
from generator import generate_answer

def rag_pipeline(user_query: str):
    # TEMP: hardcoded slug (later auto-detect via NER)
    location_slug = None

    # Retrieve relevant chunks
    chunks = get_relevant_context_hybrid(user_query, location_slug)

    if not chunks:
        return "Sorry, mujhe is query se related koi information nahi mili."

    # Build prompt
    prompt = build_prompt(chunks, user_query)

    # Generate answer
    return generate_answer(prompt)
