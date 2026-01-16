import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gen_model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(prompt: str):
    response = gen_model.generate_content(prompt)
    return response.text
