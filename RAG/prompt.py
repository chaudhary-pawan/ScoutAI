def build_prompt(context_chunks, user_query):
    context_text = "\n".join(
        [f"{i+1}. {c['content']}" for i, c in enumerate(context_chunks)]
    )

    return f"""
You are a travel assistant.

Use ONLY the information below.
If the answer is not present, say:
"I don't have enough information."

Context:
---
{context_text}
---

Question:
{user_query}

Answer clearly and concisely.
"""
