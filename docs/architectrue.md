🧠 Project Architecture — Multi-Agent Knowledge System
📘 Overview

Multi-Agent Knowledge là một hệ thống AI đa tác tử (multi-agent) hỗ trợ học tập và nghiên cứu trong lĩnh vực AI & Machine Learning.
Hệ thống có khả năng:

Tìm kiếm và tóm tắt các paper tiếng Anh mới nhất.

Giải thích khái niệm theo trình độ người học.

Cung cấp code ví dụ minh họa.

Kết hợp RAG + Multi-Agent Orchestration.

⚙️ System Architecture
🔹 Components
Component	Description
Router Agent	Phân loại intent và điều hướng yêu cầu đến agent phù hợp.
Knowledge Agent (RAG)	Tìm kiếm, truy xuất thông tin và trả lời có trích dẫn.
Explain Agent	Giải thích khái niệm theo cấp độ người học.
Code Agent	Sinh ví dụ code minh họa cho thuật toán hoặc khái niệm.
Coordinator	Điều phối workflow giữa các agent và quản lý context.
Frontend (Streamlit)	Giao diện tương tác với người dùng.
API (FastAPI)	Cầu nối giữa frontend và hệ thống agent.
🧩 Directory Structure
multi-agent-knowledge/
│
├── docs/
│   ├── architecture.md
│   └── plan_week1.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── api/
│   ├── frontend/
│   ├── agents/
│   ├── ingestion/
│   ├── embeddings/
│   ├── vectordb/
│   └── utils/
│
├── notebooks/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md

🔁 Data Flow
User Query
   ↓
Router Agent → xác định intent
   ↓
Coordinator → gọi agent phù hợp
   ↓
Knowledge Agent (RAG) ↔ Vector DB (FAISS)
   ↓
Explain Agent hoặc Code Agent (tùy task)
   ↓
Coordinator → tổng hợp kết quả
   ↓
Frontend (Streamlit UI)

🧠 Model Configuration (Hybrid Setup)
Agent	Model	Hosting
Router	DistilBERT / GPT-4o-mini	Local
Knowledge Agent	GPT-4o / Claude / Gemini	Cloud
Explain Agent	GPT-4o-mini / Gemini	Cloud
Code Agent	DeepSeek-Coder / GPT-4o-mini	Cloud
Coordinator	Python logic (LangGraph / CrewAI)	Local
📚 RAG Pipeline Overview

Ingestion

Parse PDF (PyPDF2, langchain.document_loaders)

Clean text and chunking

Embedding

Generate embeddings (SentenceTransformers / OpenAI embeddings)

Store in FAISS vector database

Retrieval & Generation

Retrieve top-k relevant chunks

Send context + query to LLM

Generate answer with citations

🧩 Agents Interaction Example

User Query:

“Explain the difference between RAG and fine-tuning with examples.”

Flow:

Router → detect intent: explain + compare

Coordinator → call Knowledge Agent to fetch papers

Knowledge Agent → retrieve and summarize

Explain Agent → produce level-based explanation

Code Agent → optional code snippet

Coordinator → combine and return final output

💡 Technologies Used

LLM Orchestration: LangChain, CrewAI, LangGraph

Embeddings & RAG: FAISS, SentenceTransformers, OpenAI API

Frontend: Streamlit

Backend: FastAPI

Data Parsing: PyPDF2

Infra: Docker

Testing: Pytest

🚀 MVP Deliverables (End of Week 1)

CLI & Streamlit demo

Multi-agent pipeline hoạt động: Router → Knowledge → Explain

Kết quả có trích dẫn từ paper

Có ví dụ code minh họa

Cấu trúc code rõ ràng, dễ mở rộng