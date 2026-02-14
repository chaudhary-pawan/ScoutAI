import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn


from RAG.rag_pipeline import rag_pipeline 

app = FastAPI(title="ScoutAI Backend")

# ==========================================
#  SECURITY SETUP 
# ==========================================
API_KEY_NAME = "x-api-key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Render Environment Variable se password uthayega
SERVER_API_KEY = os.getenv("BACKEND_API_KEY")

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    
    if not api_key_header:
        raise HTTPException(
            status_code=403, 
            detail="⛔ Access Denied: API Key missing in headers"
        )
    
    
    if not SERVER_API_KEY:
         print("⚠️ WARNING: Backend API Key not set in Environment!")
         
    
    elif api_key_header != SERVER_API_KEY:
        raise HTTPException(
            status_code=403, 
            detail="⛔ Access Denied: Invalid API Key"
        )
        
    return api_key_header
# ==========================================


class UserRequest(BaseModel):
    query: str

# 1. Health Check 
@app.get("/")
def home():
    return {"status": "Active", "message": "Server is Secured 🔒"}

# 2. Main API Endpoint (LOCKED)

@app.post("/chat", dependencies=[Depends(get_api_key)]) 
def chat_endpoint(request: UserRequest):
    user_query = request.query.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        response = rag_pipeline(user_query)
        return {"answer": response}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)