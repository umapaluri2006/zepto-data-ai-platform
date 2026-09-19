# -*- coding: utf-8 -*-
"""
support_assistant/app.py
FastAPI application exposing POST /ask endpoint for Zepto GenAI Support Assistant.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import ask_assistant, UserQuery, AgentResponse
from config import MOCK_LLM

app = FastAPI(
    title="Zepto AI Support Assistant API",
    description="Grounded GenAI Policy Support Assistant powered by LangGraph, ChromaDB, and FastAPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Zepto AI Support Assistant API",
        "mock_mode": MOCK_LLM,
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "mock_llm": MOCK_LLM}

@app.post("/ask", response_model=AgentResponse)
def ask_endpoint(query_input: UserQuery):
    """
    POST /ask endpoint:
    Accepts customer query JSON: {"query": "..."}
    Returns validated schema: {"answer": "...", "sources": ["..."], "confidence": 1.0}
    """
    if not query_input.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        response = ask_assistant(query_input.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Zepto AI Support Assistant on http://0.0.0.0:7860")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
