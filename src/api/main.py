import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel

from src.api.routers.chat_router import router as chat_router, GlobalState
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.explain_agent import ExplainAgent
from src.agents.code_agent import CodeAgent

# =========================== Logging setup ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("multi-agent-backend")

# =========================== App init ===========================
app = FastAPI(
    title="Multi-Agent Knowledge API",
    description="Backend API for multi-agent knowledge & code assistant",
    version="1.2.0"
)

# =========================== CORS ===========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================== Routers ===========================
app.include_router(chat_router, prefix="")

# =========================== Models ===========================
class QueryRequest(BaseModel):
    query: str
    agent: str = "auto"       # 'auto' or ['knowledge', 'explain', 'code']
    web_search: bool = True   # toggle web search
    mode: str = "auto"        # 'auto' | 'manual'

# =========================== Endpoints ===========================
@app.get("/")
def root():
    return {"message": "Multi-Agent Knowledge API is running."}

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"}, status_code=200)

@app.post("/route")
def route_query(request: QueryRequest):
    """
    Route query to the selected agent (manual mode).
    Auto mode default = KnowledgeAgent.
    """
    try:
        # Cập nhật trạng thái web search toàn cục
        GlobalState.web_search_enabled = request.web_search
        KnowledgeAgent.enable_web_search = request.web_search

        # Manual mode: gọi agent được chọn
        if request.mode.lower() == "manual" and request.agent in ["knowledge", "explain", "code"]:
            agent_map = {
                "knowledge": KnowledgeAgent(),
                "explain": ExplainAgent(),
                "code": CodeAgent(),
            }
            result = agent_map[request.agent].run(request.query)
            return {"intent": request.agent, "answer": result}

        # Auto mode (không còn auto detect intent nữa)
        # => mặc định gọi KnowledgeAgent
        result = KnowledgeAgent().run(request.query)
        return {"intent": "knowledge", "answer": result}

    except Exception as e:
        logger.exception(f"[ERROR] /route failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
