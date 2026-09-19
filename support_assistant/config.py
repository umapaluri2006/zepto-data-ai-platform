# -*- coding: utf-8 -*-
"""
support_assistant/config.py
Configuration parameters for Zepto GenAI Support Assistant.
"""

import os

# Base directory for support assistant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"

# MOCK_LLM toggle: Default is 1 (graded offline baseline)
# Set MOCK_LLM=0 only for optional live LLM calls (e.g. Groq / OpenAI)
MOCK_LLM = os.getenv("MOCK_LLM", "1").strip().lower() in ("1", "true", "yes")

# Optional LLM API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
