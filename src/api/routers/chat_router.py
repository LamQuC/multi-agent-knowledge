# src/api/routers/chat_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from src.agents.router import IntentRouter

router = APIRouter()
intent_router = IntentRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/chat")
async def chat(request: ChatRequest):
    return intent_router.route(request.query)
