# -*- coding: utf-8 -*-
"""
support_assistant/test_assistant.py
Test suite and verification runner for FastAPI Support Assistant API.
"""

import os
import sys
import json
from starlette.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

client = TestClient(app)

def test_endpoints():
    print("="*80)
    print("TESTING ZEPTO SUPPORT ASSISTANT FASTAPI ENDPOINTS")
    print("="*80)
    
    # 1. Health check
    res_health = client.get("/health")
    print(f"\n[GET /health] Status: {res_health.status_code}")
    print(f"Response: {res_health.json()}")
    assert res_health.status_code == 200
    
    # 2. Policy Query 1 (Delivery)
    q1 = {"query": "What is the standard delivery fee for orders below INR 149?"}
    res_q1 = client.post("/ask", json=q1)
    print(f"\n[POST /ask - Policy Query 1] Input: {q1['query']}")
    print(f"Status Code: {res_q1.status_code}")
    print(f"JSON Response:\n{json.dumps(res_q1.json(), indent=2)}")
    assert res_q1.status_code == 200
    data1 = res_q1.json()
    assert "answer" in data1 and "sources" in data1 and "confidence" in data1
    assert "doc_01" in data1["sources"]
    assert data1["confidence"] == 1.0
    
    # 3. Policy Query 2 (Refunds & Returns)
    q2 = {"query": "How many days do I have to return an unopened packaged grocery item?"}
    res_q2 = client.post("/ask", json=q2)
    print(f"\n[POST /ask - Policy Query 2] Input: {q2['query']}")
    print(f"Status Code: {res_q2.status_code}")
    print(f"JSON Response:\n{json.dumps(res_q2.json(), indent=2)}")
    assert res_q2.status_code == 200
    data2 = res_q2.json()
    assert "doc_02" in data2["sources"]
    
    # 4. Policy Query 3 (Membership Tiers)
    q3 = {"query": "What are the perks included with Zepto Pass+ membership?"}
    res_q3 = client.post("/ask", json=q3)
    print(f"\n[POST /ask - Policy Query 3] Input: {q3['query']}")
    print(f"Status Code: {res_q3.status_code}")
    print(f"JSON Response:\n{json.dumps(res_q3.json(), indent=2)}")
    assert res_q3.status_code == 200
    data3 = res_q3.json()
    assert "doc_03" in data3["sources"]
    
    # 5. General Query (Out of scope - Direct refusal)
    q4 = {"query": "What is the distance between the Earth and the Moon?"}
    res_q4 = client.post("/ask", json=q4)
    print(f"\n[POST /ask - General Query (Refusal)] Input: {q4['query']}")
    print(f"Status Code: {res_q4.status_code}")
    print(f"JSON Response:\n{json.dumps(res_q4.json(), indent=2)}")
    assert res_q4.status_code == 200
    data4 = res_q4.json()
    assert data4["sources"] == []
    assert data4["answer"] == "I can only answer questions about Zepto policies right now."
    
    print("\n" + "="*80)
    print("[SUCCESS] All FastAPI test assertions passed with 100% compliance!")
    print("="*80)

if __name__ == "__main__":
    test_endpoints()
