def build_prompt(context_chunks, user_query):
    context = []

    for i, chunk in enumerate(context_chunks):
        source = chunk["source_type"].upper()
        context.append(f"{i+1}. ({source}) {chunk['doc_content']}")

    joined_context = "\n\n".join(context)

    return f"""
You are ScoutAI, a smart travel assistant.

Use ONLY the information below.
If the answer is not present, say you don't know.

CONTEXT:
{joined_context}

USER QUESTION:
{user_query}

Answer clearly and helpfully.
"""
