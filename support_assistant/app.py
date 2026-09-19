# -*- coding: utf-8 -*-
"""
support_assistant/app.py
FastAPI application with:
- POST /ask API endpoint
- GET /docs Interactive Swagger UI
- GET / and GET /chat Interactive Web Page UI (Chat Interface)
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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

HTML_CHAT_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zepto AI Customer Support</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .chat-container { width: 100%; max-width: 720px; background: #1e293b; border-radius: 16px; border: 1px solid #334155; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); overflow: hidden; display: flex; flex-direction: column; height: 85vh; }
        .header { background: #e11d48; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
        .badge { background: rgba(255,255,255,0.2); font-size: 0.75rem; padding: 4px 8px; border-radius: 12px; }
        .messages { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
        .msg { max-width: 80%; padding: 14px 18px; border-radius: 12px; line-height: 1.5; font-size: 0.95rem; }
        .user-msg { background: #3b82f6; align-self: flex-end; border-bottom-right-radius: 2px; }
        .bot-msg { background: #334155; align-self: flex-start; border-bottom-left-radius: 2px; }
        .source-tag { display: inline-block; background: #0f172a; color: #38bdf8; font-size: 0.75rem; padding: 2px 8px; border-radius: 6px; margin-top: 8px; font-weight: 600; }
        .input-area { padding: 16px 20px; background: #0f172a; display: flex; gap: 12px; border-top: 1px solid #334155; }
        input { flex: 1; padding: 14px 18px; border-radius: 10px; border: 1px solid #475569; background: #1e293b; color: white; outline: none; font-size: 0.95rem; }
        input:focus { border-color: #e11d48; }
        button { background: #e11d48; color: white; border: none; padding: 14px 24px; border-radius: 10px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #be123c; }
        .quick-chips { display: flex; gap: 8px; flex-wrap: wrap; padding: 10px 24px; background: #1e293b; border-bottom: 1px solid #334155; }
        .chip { background: #334155; color: #cbd5e1; font-size: 0.8rem; padding: 6px 12px; border-radius: 20px; cursor: pointer; border: 1px solid #475569; }
        .chip:hover { background: #e11d48; color: white; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <h1>⚡ Zepto Support Assistant</h1>
            <span class="badge">LangGraph + RAG Active</span>
        </div>
        <div class="quick-chips">
            <span class="chip" onclick="ask('What is the delivery fee for orders below 149 rupees?')">Delivery Fee</span>
            <span class="chip" onclick="ask('How many days to return unopened items?')">Return Policy</span>
            <span class="chip" onclick="ask('What are Zepto Pass+ benefits?')">Zepto Pass+</span>
            <span class="chip" onclick="ask('What if rider is stuck for 20 mins?')">Rider Tracking</span>
        </div>
        <div class="messages" id="messages">
            <div class="msg bot-msg">
                Hello! 👋 I am Zepto's AI Support Assistant. Ask me anything about our delivery, returns, refunds, Zepto Pass, tracking, or cancellation policies!
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask a question about Zepto policies..." onkeydown="if(event.key==='Enter') send()">
            <button onclick="send()">Send</button>
        </div>
    </div>

    <script>
        async function send() {
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';
            appendMsg(text, 'user-msg');
            
            const typingId = appendMsg('Searching policy documents...', 'bot-msg');
            
            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: text })
                });
                const data = await res.json();
                document.getElementById(typingId).remove();
                
                let sourceHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourceHtml = '<br>' + data.sources.map(s => `<span class="source-tag">📄 Source: ${s}</span>`).join(' ');
                }
                appendMsg(data.answer + sourceHtml, 'bot-msg');
            } catch (err) {
                document.getElementById(typingId).remove();
                appendMsg('Error connecting to support service.', 'bot-msg');
            }
        }
        
        function ask(q) {
            document.getElementById('userInput').value = q;
            send();
        }
        
        function appendMsg(html, cls) {
            const id = 'msg_' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = 'msg ' + cls;
            div.innerHTML = html;
            const container = document.getElementById('messages');
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return id;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def serve_chat_ui():
    """Serves the interactive Zepto Customer Support Web Page UI."""
    return HTMLResponse(content=HTML_CHAT_UI, status_code=200)

@app.get("/health")
def health_check():
    return {"status": "healthy", "mock_llm": MOCK_LLM}

@app.post("/ask", response_model=AgentResponse)
def ask_endpoint(query_input: UserQuery):
    if not query_input.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        response = ask_assistant(query_input.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Zepto Support Assistant Web Application on http://localhost:7860")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
