# src/utils/config_loader.py
"""
Đọc cấu hình từ .env / fallback values.
Giúp quản lý model names per agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM keys / models
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    # model names (tùy bạn đổi .env hoặc ở đây)
    MODEL_CODE = os.getenv("MODEL_CODE", "gemini-2.5-flash")    # recommended code-tuned model
    MODEL_EXPLAIN = os.getenv("MODEL_EXPLAIN", "gemini-2.5-flash") # explain / general reasoning
    MODEL_KNOWLEDGE = os.getenv("MODEL_KNOWLEDGE", "gemini-2.5-flash")# retrieval / long answers

    # FAISS / embeddings
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/processed/faiss_index")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Memory thresholds
    SHORT_MEMORY_MAX = int(os.getenv("SHORT_MEMORY_MAX", "5"))
    LONG_MEMORY_PUSH_THRESHOLD = int(os.getenv("LONG_MEM_THRESHOLD", "8"))

config = Config()
