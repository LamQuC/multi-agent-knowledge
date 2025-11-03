# src/api/main.py
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from src.api.routers.chat_router import router as chat_router

# =========================
# 🚀 Logging Configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("multi-agent-backend")

# =========================
# 🌐 FastAPI App Init
# =========================
app = FastAPI(
    title="Multi-Agent Knowledge API",
    description="Backend API for multi-agent knowledge & code assistant",
    version="1.0.0"
)

# Allow frontend & external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register chat router
app.include_router(chat_router, prefix="")

# =========================
# 🏠 Basic Root Endpoint
# =========================
@app.get("/")
def root():
    return {"message": "Multi-Agent Knowledge API is running."}

# =========================
# ❤️ Health Check Endpoint
# =========================
@app.get("/health")
def health():
    """
    Simple health check — can be used by Docker, monitoring tools, or uptime checks.
    """
    details = {
        "status": "ok",
        "env": os.getenv("ENV", "development"),
        "service": "multi-agent-backend"
    }
    logger.info("Health check passed.")
    return JSONResponse(content=details, status_code=200)

# =========================
# ⚙️ Readiness Check Endpoint
# =========================
@app.get("/ready")
def ready():
    """
    Readiness probe — verify that required environment variables exist.
    """
    required_envs = ["MODEL_CODE", "MODEL_EXPLAIN"]
    missing = [var for var in required_envs if not os.getenv(var)]
    ready_status = len(missing) == 0

    details = {
        "ready": ready_status,
        "missing_env": missing,
        "service": "multi-agent-backend"
    }

    status_code = 200 if ready_status else 503
    logger.info(f"Readiness check: {details}")
    return JSONResponse(content=details, status_code=status_code)
