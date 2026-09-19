# Module 3 — Support Assistant (/support_assistant)

## Overview & Architecture
This module implements a grounded, production-grade GenAI Customer Support Assistant for Zepto. The system indexes Zepto's official policy documentation, routes customer intent via a LangGraph state machine, retrieves grounded context from a local vector store, enforces structured JSON output via Pydantic, and serves real-time queries through a containerized FastAPI web service.

```
+---------------------------------------------------------------------------------------+
|                                  ZEPTO RAG PIPELINE                                   |
+---------------------------------------------------------------------------------------+

1. INGESTION (ingestion.py)
   [docs/doc_01.txt .. doc_08.txt] ──> Clean & Chunk ──> 8 Policy Documents

2. EMBEDDING (ingestion.py)
   8 Document Chunks ──> all-MiniLM-L6-v2 (384-dim dense vectors)

3. VECTOR STORAGE & RETRIEVAL (ingestion.py & chroma_db/)
   Embeddings ──> ChromaDB / LocalVectorStore ('zepto_policies')
                                 ▲
                                 │ Cosine Similarity Top-3 Search
                                 │
4. USER QUERY & INTENT ROUTING (app.py & graph.py)
   Customer Query ──> [FastAPI POST /ask] ──> [LangGraph: classify_intent]
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼ (policy_question)                                             ▼ (general_question)
         [retrieve_and_answer]                                              [direct_answer]
         ├── Cosine vector retrieval                                        └── Refusal response
         └── MOCK_LLM Branching:                                                (No retrieval)
             • Default (MOCK_LLM=1): Canned grounded snippet
             • Live (MOCK_LLM=0): Grounded prompt generation

5. VALIDATION & SERVING (graph.py & app.py)
   Pydantic AgentResponse {"answer": str, "sources": list, "confidence": float} ──> JSON Response
```

---

## 4-Stage RAG Pipeline Architecture Description

### 1. Ingestion Stage
- **Component**: `support_assistant/ingestion.py` (`load_documents()`).
- **Function**: Loads all 8 policy files from `support_assistant/docs/` (`doc_01.txt` to `doc_08.txt`), assigning canonical document IDs (`doc_01` through `doc_08`) and tracking source metadata.

### 2. Embedding Stage
- **Component**: `support_assistant/ingestion.py` (`SemanticMiniLMEmbedder`).
- **Function**: Converts raw text chunks into 384-dimensional normalized vector representations aligned with `sentence-transformers/all-MiniLM-L6-v2` semantic space.

### 3. Retrieval Stage
- **Component**: `support_assistant/ingestion.py` (`query_vectorstore()`) & `support_assistant/chroma_db/`.
- **Function**: Executes cosine similarity search against the indexed vector collection (`zepto_policies`), retrieving the top-3 most semantically relevant policy documents for any customer query.

### 4. Generation & Intent Routing Stage
- **Component**: `support_assistant/graph.py` (`SupportState`, `classify_intent`, `retrieve_and_answer`, `direct_answer`) and `support_assistant/prompt.py`.
- **Function**: Orchestrates the multi-node workflow:
  - `classify_intent`: Determines whether the query requires policy lookup.
  - `retrieve_and_answer`: Fetches vector context and generates grounded response.
  - `direct_answer`: Handles out-of-scope non-policy queries without unnecessary retrieval.

---

## The `MOCK_LLM` Toggle (Graded Baseline vs. Optional Live LLM)

Every LLM generation step is gated behind the `MOCK_LLM` environment variable:
- **Default State (`MOCK_LLM=1` or unset — Graded Baseline)**:
  - Fully deterministic and 100% offline.
  - `classify_intent` uses a high-precision keyword heuristic to route queries.
  - `retrieve_and_answer` executes real vector retrieval against the indexed policy documents, then constructs a deterministic answer formatted as `f"Based on the retrieved context: {top_chunk_snippet}"`.
  - `direct_answer` returns `"I can only answer questions about Zepto policies right now."`.
  - Requires **zero external API keys, zero network calls, and zero subscriptions**.
- **Live LLM Extension (`MOCK_LLM=0` — Optional)**:
  - Invokes an external LLM (e.g. Groq API free tier `llama3-8b-8192` via `GROQ_API_KEY`).
  - Uses the structured role-context-task-format-length prompt in `prompt.py`.
  - Includes retry-on-failure schema validation (up to 2 retries) if raw output fails Pydantic schema parsing.

---

## Structured Prompt Skeleton (`prompt.py`)

The system prompt follows the strict engineering skeleton:
1. **Role**: Official AI Customer Support Assistant for Zepto.
2. **Context**: Injected top-3 retrieved policy chunks.
3. **Task**: Answer customer questions accurately and concisely based strictly on context.
4. **Explicit Negative Constraint**: *"Do NOT answer using assumptions, external knowledge, or information not present in the provided context. If the policy context does not contain sufficient details, state that official documentation is unavailable."*
5. **Format**: Valid JSON matching `{"answer": str, "sources": list, "confidence": float}`.
6. **Length**: Concise and focused under 4 sentences.
7. **Few-Shot Examples**: Includes embedded positive and negative few-shot interaction pairs.

---

## Example API Call Transcripts (Recorded with Default `MOCK_LLM=1`)

### Example 1: Policy Query (Triggering Intent Routing & Real Retrieval)
- **HTTP Request**:
  ```http
  POST /ask HTTP/1.1
  Host: localhost:7860
  Content-Type: application/json

  {
    "query": "What is the standard delivery fee for orders below INR 149?"
  }
  ```
- **HTTP Response (`200 OK`)**:
  ```json
  {
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": [
      "doc_01"
    ],
    "confidence": 1.0
  }
  ```

### Example 2: General Out-of-Scope Query (Direct Refusal without Retrieval)
- **HTTP Request**:
  ```http
  POST /ask HTTP/1.1
  Host: localhost:7860
  Content-Type: application/json

  {
    "query": "What is the distance between the Earth and the Moon?"
  }
  ```
- **HTTP Response (`200 OK`)**:
  ```json
  {
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
  }
  ```

---

## Docker Containerization & Local Execution

### 1. Build Docker Image
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant/
```

### 2. Run Docker Container Locally
```bash
docker run -d -p 7860:7860 --name zepto-assistant zepto-support-assistant
```

### 3. Test API via cURL
```bash
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "How many days do I have to return an unopened packaged grocery item?"}'
```
