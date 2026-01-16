import os
import google.generativeai as genai
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# ==================================================
# CONFIGURATION
# ==================================================
SUPABASE_URL = "https://qwozjwjqhgqrgvjadrgm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF3b3pqd2pxaGdxcmd2amFkcmdtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NzA4ODA4NywiZXhwIjoyMDgyNjY0MDg3fQ.M3oiU4pMfsZdgE6vA7YXlrIwSfk_y5yrpD6V2jw0Qs0"
GEMINI_API_KEY = "AIzaSyBD-rye1ryz8XkHUWQRDRALdRSDIz1dPcA" # <-- Yahan wo key paste karna jo abhi mili

# 1. Clients Setup
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Gemini Setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') # Free aur fast model

# ==================================================
# STEP 1: RETRIEVAL (Dhoondhne wala function)
# ==================================================
def get_relevant_context(user_query):
    print(f"\n🔍 Searching for: '{user_query}'...")
    
    # Query Embedding
    query_vector = embedding_model.encode(user_query).tolist()
    
    # Supabase Call
    response = supabase.rpc(
        "match_all_documents", 
        {
            "query_embedding": query_vector,
            "match_threshold": 0.30,
            "match_count": 6
        }
    ).execute()
    
    matches = response.data
    if not matches:
        return None
    
    # Results Formatting
    # Function: get_relevant_context ke andar...

    formatted_results = []
    print(f"✅ Found {len(matches)} relevant matches.")
    
    for i, item in enumerate(matches):
        source = item.get('source_type', 'Unknown').upper()
        
        # --- YE WALI LINE CHANGE KARNI HAI ---
        # Pehle: content = item.get('content', '')
        # Ab:
        content = item.get('doc_content', '')   # <--- Change this to 'doc_content'
        
        formatted_results.append(f"--- Info {i+1} ({source}) ---\n{content}")
        
    return "\n\n".join(formatted_results)

# ==================================================
# STEP 2: GENERATION (Gemini Answer)
# ==================================================
def ask_gemini(user_query):
    # Context lao
    context = get_relevant_context(user_query)
    
    if not context:
        return "Sorry, mujhe database mein isse related koi info nahi mili."

    # Prompt Engineering for Gemini
    prompt = f"""
    You are 'ScoutAI', a smart travel assistant for Scoutripper.
    Answer the user's question mostly based on the provided Context below.
    
    CONTEXT INFORMATION:
    {context}
    
    USER QUESTION: 
    {user_query}
    
    INSTRUCTIONS:
    - Keep the tone helpful and friendly (Hinglish or English as per query).
    - If the context has specific details (price, distance), use them accurately.
    - If answer is not in context, say you don't know.
    """
    
    print("🤖 Gemini is thinking...")
    
    # Gemini Call
    response = model.generate_content(prompt)
    
    return response.text

# ==================================================
# TESTING
# ==================================================
if __name__ == "__main__":
    # Yahan apna sawal change karke test karna
    question = "Tell me about Trek Clean up drives."
    
    answer = ask_gemini(question)
    print("\n" + "="*50)
    print("✨ GEMINI ANSWER:")
    print("="*50)
    print(answer)