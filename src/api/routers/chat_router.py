# src/api/routers/chat_router.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.router import IntentRouter
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.code_agent import CodeAgent
from src.agents.explain_agent import ExplainAgent

router = APIRouter()
intent_router = IntentRouter()

# --- Lưu trạng thái toggle web search ---
class GlobalState:
    web_search_enabled = True


class ChatRequest(BaseModel):
    query: str
    agent: str = "auto"  # "auto", "knowledge", "code", "explain"
    web_search: bool | None = None  # None = giữ nguyên trạng toggle hiện tại


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Cập nhật toggle nếu user gửi web_search
        if request.web_search is not None:
            GlobalState.web_search_enabled = request.web_search

        query = request.query.strip()

        # --- Manual chọn agent ---
        if request.agent in ["knowledge", "code", "explain"]:
            if request.agent == "knowledge":
                KnowledgeAgent.enable_web_search = GlobalState.web_search_enabled
                agent = KnowledgeAgent()
            elif request.agent == "code":
                agent = CodeAgent()
            else:
                agent = ExplainAgent()

            result = agent.run(query)
            return JSONResponse(
                content={
                    "answer": result.get("answer", ""),
                    "intent": request.agent,
                    "source": result.get("source", "manual"),
                }
            )

        # --- Auto detect intent ---
        else:
            result = intent_router.route(query)
            # đảm bảo trả JSON đúng chuẩn
            if isinstance(result, dict):
                return JSONResponse(content=result)
            return JSONResponse(content={"answer": str(result)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/toggle_websearch")
def toggle_websearch(enable: bool):
    """API riêng cho frontend toggle websearch."""
    GlobalState.web_search_enabled = enable
    KnowledgeAgent.enable_web_search = enable
    return {"status": "ok", "enabled": enable}
