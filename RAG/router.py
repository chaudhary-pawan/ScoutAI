import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

router_model = genai.GenerativeModel("gemini-2.5-flash")

def route_query(query: str) -> str:
    prompt = f"""
Classify the user query into ONE category:
- locations
- treks
- experiences
- multiple

Return ONLY one word.

Query:
{query}
"""
    response = router_model.generate_content(prompt)
    return response.text.strip().lower()
