# src/utils/llm_manager.py
"""
LLM Manager: tạo wrapper LLM (LangChain ChatGoogleGenerativeAI) theo model name.
Sử dụng langchain-google-genai connector.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.config_loader import config


def create_langchain_llm(model_name: str = None, temperature: float = 0.0):
    """
    Trả về LangChain LLM wrapper cho Gemini model.
    """
    m = model_name or config.MODEL_EXPLAIN

    # Lấy API key từ .env hoặc config
    api_key = os.getenv("GOOGLE_API_KEY") or config.GEMINI_API_KEY
    if not api_key:
        raise ValueError("⚠️ Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment/config.")

    # Truyền trực tiếp API key vào model (bắt buộc để tránh ADC mode)
    return ChatGoogleGenerativeAI(
        model=m,
        temperature=temperature,
        api_key=api_key
    )
