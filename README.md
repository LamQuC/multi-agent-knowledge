# Multi Agent Knowledge Chat

## Table of Contents
- [Overview](#overview)
- [Architecture and components](#architecture-and-components)
- [File structure](#file-structure)
- [Requirements and environment](#requirements-and-environment)
- [Setup and local run with Docker](#setup-and-local-run-with-docker)
- [Running without rebuild after code changes](#running-without-rebuild-after-code-changes)
- [How the system works end to end](#how-the-system-works-end-to-end)
- [API specification and examples](#api-specification-and-examples)
- [Frontend usage and behaviors](#frontend-usage-and-behaviors)
- [Knowledge agent behavior and toggle web search logic](#knowledge-agent-behavior-and-toggle-web-search-logic)
- [How to extend with new tools or agents](#how-to-extend-with-new-tools-or-agents)
- [FAISS index and embedding pipeline](#faiss-index-and-embedding-pipeline)
- [Memory and conversation sessions](#memory-and-conversation-sessions)
- [Debugging and common issues](#debugging-and-common-issues)
- [Docker disk cleanup and best practices](#docker-disk-cleanup-and-best-practices)
- [Testing and CI suggestions](#testing-and-ci-suggestions)
- [Security notes and operational tips](#security-notes-and-operational-tips)

---

## 1. Overview

This repository implements a multi-agent assistant with three main agents named **knowledge**, **explain**, and **code**.  
The backend is a **FastAPI** service.  
The frontend is a **Streamlit single-page chat app**.

The knowledge agent supports two modes:
- **Web search ON**: Uses DuckDuckGo to fetch live web results.
- **Web search OFF**: Uses FAISS retrieval and the LLM.

Chat sessions are persisted on disk so users can re-open past conversations and continue.

This README documents:
- Full setup from zero to production Docker Compose.
- Local development workflow without rebuilds.
- How to extend with new tools or agents.

---

## 2. Architecture and components

### Backend
- **FastAPI-based HTTP API**
- Chat router for manual and auto routing
- Intent router for auto classification
- Three agents: `KnowledgeAgent`, `ExplainAgent`, `CodeAgent`
- FAISS vector store for retrieval
- Memory managers for short and long-term memory

### Frontend
- Streamlit UI storing sessions as JSON files
- Toggle for web search (only when `knowledge` agent selected)
- Supports Auto and Manual modes
- Create, list, open, delete chat sessions

### Tools
- `DuckDuckGoSearchRun` for web search
- LangChain LLM factory (`create_langchain_llm`)
- Sentence-transformers + FAISS for embeddings

### Deployment
- Docker images for backend and frontend
- Docker Compose defines services, ports, and volumes
- Bind mounts for live code editing without rebuild

---

## 3. File structure

multi-agent-knowledge
├─ docker-compose.yml
├─ backend
│ ├─ Dockerfile
│ ├─ requirements.txt
│ └─ src
│ ├─ api
│ │ ├─ main.py
│ │ └─ routers
│ │ └─ chat_router.py
│ ├─ agents
│ │ ├─ base_agent.py
│ │ ├─ router.py
│ │ ├─ knowledge_agent.py
│ │ ├─ explain_agent.py
│ │ └─ code_agent.py
│ ├─ vectordb
│ │ └─ faiss_index.py
│ ├─ agents
│ │ └─ memory
│ │ ├─ memory_manager.py
│ │ └─ long_term_memory.py
│ └─ utils
│ ├─ llm_manager.py
│ ├─ config_loader.py
│ └─ llm_client.py
├─ frontend
│ ├─ Dockerfile
│ ├─ app.py
│ └─ data
│ └─ chats
└─ README.md

markdown
Copy code

**Important files:**
- `main.py`: Registers routes.
- `chat_router.py`: Handles `/chat` and web toggle.
- `knowledge_agent.py`: Implements FAISS + web search logic.
- `faiss_index.py`: Builds and loads vector store.
- `memory_manager.py`: Handles conversation persistence.
- `frontend/app.py`: Streamlit UI logic.

---

## 4. Requirements and environment

### Python Version
Python 3.11+

### Core Dependencies
langchain==0.3.7
langchain-core==0.3.17
langchain-community==0.3.7
langchain-google-genai==2.0.5
openai>=1.12.0
tiktoken>=0.7.0
faiss-cpu
sentence-transformers
transformers
torch
pandas
numpy<2.0.0
pypdf
requests
python-dotenv
huggingface-hub
langchain-huggingface
fastapi
uvicorn
streamlit
pytest

shell
Copy code

### Environment Variables (`.env`)
OPENAI_API_KEY=<your_api_key>
MODEL_CODE=gpt-4
MODEL_EXPLAIN=gpt-4
MODEL_KNOWLEDGE=gpt-4
FAISS_INDEX_PATH=./data/faiss.index
SHORT_MEMORY_MAX=10
LONG_MEMORY_PUSH_THRESHOLD=30

yaml
Copy code

---

## 5. Setup and local run with Docker

### Steps
1. Clone repository  
2. Create `.env` file with LLM keys and model names  
3. Build and run containers:

docker compose up -d --build

yaml
Copy code

Subsequent runs:
docker compose up -d

sql
Copy code

View logs:
docker compose logs -f backend
docker compose logs -f frontend

vbnet
Copy code

Stop containers:
docker compose down

yaml
Copy code

Run locally outside Docker:

Backend:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

makefile
Copy code

Frontend:
streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0

yaml
Copy code

---

## 6. Running without rebuild after code changes

Bind mount code for live editing:
```yaml
services:
  backend:
    build: ./backend
    volumes:
      - ./backend/src:/app/src
    ports:
      - "8000:8000"
Restart without rebuild:

nginx
Copy code
docker compose restart backend
Or use auto reload in development:

nginx
Copy code
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
7. How the system works end to end
Manual mode + web search ON:

Frontend sends web_search=true

Backend calls DuckDuckGoSearchRun

LLM summarizes and answers

Memory updated

Manual mode + web search OFF:

FAISS retrieval first

Fallback to LLM if no results

Auto mode:

Intent router classifies → agent executes

8. API specification and examples
Base URL: http://localhost:8000

Health:

css
Copy code
GET /
Response: {"message":"Multi-Agent Knowledge API is running."}
Chat:

bash
Copy code
POST /chat
{
  "query": "Your question text",
  "agent": "knowledge",
  "web_search": true
}
Response:

json
Copy code
{
  "answer": "Assistant response text",
  "intent": "knowledge",
  "source": "web"
}
Toggle Web Search:

bash
Copy code
POST /toggle_websearch?enable=true
Response: {"status":"ok","enabled":true}
9. Frontend usage and behaviors
Mode selector: Auto / Manual

Web search toggle visible only in manual mode (knowledge agent)

Sessions saved in frontend/data/chats

Chat bubble interface

Session example:

json
Copy code
{
  "title":"Chat 2025-11-09_16-00-00",
  "created_at":"2025-11-09_16-00-00",
  "messages":[
    {"role":"user","content":"Hello"},
    {"role":"assistant","content":"Hi there"}
  ]
}
10. Knowledge agent behavior and toggle logic
Toggle	Behavior
ON	Uses DuckDuckGoSearchRun + LLM synthesis
OFF	Uses FAISS retrieval chain or fallback LLM

Memory rules:

Short memory: stores last few turns

Long memory: pushed when threshold exceeded

11. How to extend with new tools or agents
New Tool

Add method in knowledge_agent.py

Wrap in error handling

Test with isolated inputs

New Agent

Create src/agents/new_agent.py

Subclass AgentBase

Map in router and frontend dropdown

Best Practices

Pure functions

Small, testable modules

Unit tests for each

12. FAISS index and embedding pipeline
Build FAISS Index

bash
Copy code
python src/vectordb/build_index.py --docs ./data/docs --index-path ./data/faiss.index
Steps

Load and split documents

Compute embeddings

Save FAISS index

Used automatically by KnowledgeAgent for retrieval.

13. Memory and conversation sessions
Short memory: Keeps last few exchanges

Long memory: Stores persistent context

Frontend sessions: Saved as JSON files per conversation

14. Debugging and common issues
Issue	Fix
LangChain warnings	Align package versions
DuckDuckGo empty results	Install ddgs, check container internet
Streamlit errors	Update version or rerun
404 /chat	Check backend URL and network alias
ResourceWarning	Always use with open()

15. Docker disk cleanup and best practices
Free up space:

css
Copy code
docker system prune -a --volumes
Remove unused images:

arduino
Copy code
docker image rm <image_id>
Tips

Use --no-cache-dir

Use small base images

Bind mounts for dev mode

16. Testing and CI suggestions
Use pytest for unit + integration tests

Mock LLM calls for CI

Run static security scans

Add memory + FAISS retrieval test cases

17. Security notes and operational tips
Never commit .env or API keys

Use API rate limits

Sanitize uploaded docs

Protect UI/backend with auth

Quick start checklist

bash
Copy code
git clone <repo>
cd multi-agent-knowledge
cp .env.example .env
docker compose up -d --build
Test endpoints:

Frontend: http://localhost:8501

Backend: http://localhost:8000/health