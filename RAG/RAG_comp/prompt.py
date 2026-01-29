def build_prompt(chunks, user_query):
    if not chunks:
        return f"""
User Question:
{user_query}

No relevant context was found.
Please respond that you don't know.
"""

    formatted_chunks = []

    for i, chunk in enumerate(chunks):
        source = chunk["source_type"].upper()
        formatted_chunks.append(
            f"{i+1}. ({source}) {chunk['doc_content']}"
        )

    context_text = "\n\n".join(formatted_chunks)

    return f"""
You are ScoutAI, a smart travel assistant.

Use ONLY the information below.
If the answer is not present, say you don't know.

CONTEXT:
{context_text}

USER QUESTION:
{user_query}

Answer clearly and helpfully.
"""
