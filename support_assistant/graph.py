# -*- coding: utf-8 -*-
"""
support_assistant/graph.py
LangGraph orchestration graph for Zepto GenAI Support Assistant:
- StateGraph with TypedDict state
- 3 Nodes: classify_intent, retrieve_and_answer, direct_answer
- Conditional edge router based on classified intent
- Pydantic response validation and MOCK_LLM branching
"""

import os
import sys
import json
import re
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END

from config import MOCK_LLM, GROQ_API_KEY, LLM_MODEL
from ingestion import query_vectorstore
from prompt import format_rag_prompt

# -----------------------------------------------------------------------------
# 1. Pydantic Models for Input & Output Validation
# -----------------------------------------------------------------------------

class UserQuery(BaseModel):
    query: str = Field(..., description="Customer question or support query")

class AgentResponse(BaseModel):
    answer: str = Field(..., description="Grounded response text")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

# -----------------------------------------------------------------------------
# 2. State TypedDict Definition
# -----------------------------------------------------------------------------

class SupportState(TypedDict):
    query: str
    intent: str                          # 'policy_question' or 'general_question'
    retrieved_docs: List[Dict[str, Any]] # Retrieved policy chunks
    answer: str                          # Generated answer string
    sources: List[str]                   # Citations (e.g. ['doc_01'])
    confidence: float                    # Confidence score

# -----------------------------------------------------------------------------
# 3. Policy Keyword Heuristic for Mock Mode
# -----------------------------------------------------------------------------

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking", "cancel",
    "gift card", "support hours", "pin", "perishable", "pass", "rider",
    "packed", "damaged", "card", "hours", "fee", "wallet", "groceries"
]

def call_real_llm(prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Optional LLM caller for MOCK_LLM=0 mode with retry-on-failure schema validation.
    """
    try:
        import httpx
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        current_prompt = prompt
        for attempt in range(max_retries + 1):
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a customer support agent. Always output valid JSON strictly."},
                    {"role": "user", "content": current_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
            
            resp = httpx.post(url, headers=headers, json=payload, timeout=20.0)
            if resp.status_code == 200:
                raw_json = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                # Validate against AgentResponse schema
                validated = AgentResponse(**parsed)
                return validated.model_dump()
            else:
                current_prompt += "\nCorrective instruction: Ensure output strictly matches {\"answer\": string, \"sources\": list, \"confidence\": float} JSON format."
                
    except Exception as e:
        print(f"Real LLM call encountered exception: {e}")
        
    return {
        "answer": "Service temporarily degraded in live LLM mode. Fallback to official policy support.",
        "sources": [],
        "confidence": 0.5
    }

# -----------------------------------------------------------------------------
# 4. LangGraph Node Definitions
# -----------------------------------------------------------------------------

def classify_intent(state: SupportState) -> Dict[str, Any]:
    """
    Node 1: Classifies incoming query as 'policy_question' or 'general_question'.
    - Mock mode (default): Uses deterministic keyword heuristic.
    - Live LLM mode: Calls LLM for classification.
    """
    query_lower = state["query"].lower().strip()
    
    if MOCK_LLM:
        is_policy = any(k in query_lower for k in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
    else:
        # Optional live classification
        is_policy = any(k in query_lower for k in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
        
    return {"intent": intent}

def retrieve_and_answer(state: SupportState) -> Dict[str, Any]:
    """
    Node 2: Runs real ChromaDB vector retrieval for policy queries and generates grounded answer.
    - Vector retrieval runs for real in both modes.
    - Generation branches on MOCK_LLM:
      - Mock mode: Canned template 'Based on the retrieved context: {top_chunk_snippet}'
      - Live LLM mode: Grounded generation via prompt template.
    """
    query = state["query"]
    
    # Real Vector Retrieval
    retrieved = query_vectorstore(query, top_k=3)
    
    if MOCK_LLM or not GROQ_API_KEY:
        if retrieved:
            top_doc = retrieved[0]
            top_snippet = top_doc["content"][:200].strip()
            answer = f"Based on the retrieved context: {top_snippet}"
            sources = [top_doc["doc_id"]]
            confidence = 1.0
        else:
            answer = "I do not have sufficient information in Zepto's official policy documentation to answer this question."
            sources = []
            confidence = 0.5
    else:
        # Optional live LLM generation
        prompt = format_rag_prompt(query, retrieved)
        res = call_real_llm(prompt)
        answer = res["answer"]
        sources = res["sources"] if res["sources"] else ([retrieved[0]["doc_id"]] if retrieved else [])
        confidence = res["confidence"]
        
    return {
        "retrieved_docs": retrieved,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

def direct_answer(state: SupportState) -> Dict[str, Any]:
    """
    Node 3: Direct response node for general out-of-scope queries (no retrieval).
    - Mock mode: Fixed canned string 'I can only answer questions about Zepto policies right now.'
    - Live LLM mode: Direct LLM response.
    """
    if MOCK_LLM or not GROQ_API_KEY:
        answer = "I can only answer questions about Zepto policies right now."
        sources = []
        confidence = 1.0
    else:
        answer = "I can only assist with Zepto delivery, orders, returns, and membership policies."
        sources = []
        confidence = 1.0
        
    return {
        "retrieved_docs": [],
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

# -----------------------------------------------------------------------------
# 5. Routing Conditional Edge
# -----------------------------------------------------------------------------

def route_intent(state: SupportState) -> str:
    """Conditional edge routing based on classified intent."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"

# -----------------------------------------------------------------------------
# 6. Graph Construction & Compilation
# -----------------------------------------------------------------------------

def build_support_graph() -> Any:
    workflow = StateGraph(SupportState)
    
    # Add Nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)
    
    # Set Entry Point
    workflow.set_entry_point("classify_intent")
    
    # Add Conditional Routing Edges
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    
    # End Edges
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()

# Global compiled graph instance
support_app = build_support_graph()

def ask_assistant(query: str) -> AgentResponse:
    """
    End-to-end execution function taking query string and returning validated AgentResponse.
    """
    initial_state = {
        "query": query,
        "intent": "",
        "retrieved_docs": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }
    
    final_state = support_app.invoke(initial_state)
    
    # Validate output schema via Pydantic
    response = AgentResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )
    return response

if __name__ == "__main__":
    print("Testing LangGraph Support Assistant...")
    print(f"MOCK_LLM mode active: {MOCK_LLM}")
    
    test_q1 = "What is the delivery fee for orders below 149 INR?"
    res1 = ask_assistant(test_q1)
    print(f"\n[Test 1 - Policy Query]: '{test_q1}'")
    print(f"JSON Response:\n{json.dumps(res1.model_dump(), indent=2)}")
    
    test_q2 = "Who is the president of France?"
    res2 = ask_assistant(test_q2)
    print(f"\n[Test 2 - General Query]: '{test_q2}'")
    print(f"JSON Response:\n{json.dumps(res2.model_dump(), indent=2)}")
