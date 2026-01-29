from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from RAG.rag_pipeline import rag_pipeline 

# App start
app = FastAPI(title="ScoutAI Backend")

# Input ka dhabba (Schema)
class UserRequest(BaseModel):
    query: str

# 1. Health Check (Bas check karne ke liye ki server zinda hai)
@app.get("/")
def home():
    return {"status": "Active", "message": "Server is Running.🚀"}

# 2. Main API Endpoint
@app.post("/chat")
def chat_endpoint(request: UserRequest):
    user_query = request.query.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # --- LOGIC CALL ---
        # Yahan hum aapke original code ko input de rahe hain
        # Agar rag_pipeline.py mein koi error nahi hai, toh ye perfectly chalega
        response = rag_pipeline(user_query)
        
        return {"answer": response}
    
    except Exception as e:
        return {"error": str(e)}

# Local run ke liye
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)