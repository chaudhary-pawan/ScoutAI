import openai

def route_query(query: str):
    prompt = f"""
Classify the user query into one of the following:
- locations
- treks
- experiences
- multiple

Return ONLY JSON.

Query: "{query}"
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return eval(response.choices[0].message.content)["route"]
