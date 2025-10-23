📅 Week 1 Development Plan — Multi-Agent Knowledge System
🎯 Objective

Hoàn thiện MVP phiên bản đầu tiên của hệ thống multi-agent AI có khả năng:

Phân loại intent người dùng

Thực hiện truy vấn kiến thức (RAG)

Giải thích theo trình độ người học

Hiển thị kết quả qua Streamlit UI

🗓️ Timeline
Day 1 — Project Initialization

Xác định mục tiêu và domain.

Thiết kế kiến trúc tổng thể (agents, dataflow, model).

Khởi tạo cấu trúc thư mục multi-agent-knowledge/.

Viết file:

docs/architecture.md

docs/plan_week1.md

✅ Deliverable:
Tài liệu kiến trúc & roadmap.

Day 2 — Intent Mapping & Prompt Design

Định nghĩa intent types: explain, summarize, compare, code, research.

Viết config/intents.yaml.

Tạo prompt templates cho từng agent.

Viết class cơ sở BaseAgent (src/agents/base_agent.py).

✅ Deliverable:
Intent mapping + prompt templates cơ bản.

Day 3 — Data Ingestion Pipeline

Tải 10–20 paper tiếng Anh từ ArXiv / HuggingFace.

Parse PDF → text (pdf_parsing.py).

Chunk text theo độ dài (500–1000 tokens) (chunking.py).

Lưu output trong data/processed/.

✅ Deliverable:
Dữ liệu sạch, chia nhỏ, sẵn sàng embedding.

Day 4 — Embedding & Vector Database

Sinh embeddings (embedding_builder.py) bằng SentenceTransformers / OpenAI.

Xây FAISS index (faiss_index.py).

Test truy vấn vector cơ bản.

✅ Deliverable:
Vector database hoạt động.

Day 5 — RAG Pipeline Testing

Viết hàm pipeline: query → retrieve → LLM → answer.

Test với 3 câu hỏi thực tế.

Lưu log và output mẫu.

✅ Deliverable:
RAG hoạt động ổn định, có log.

Day 6 — Multi-Agent Integration

Implement Router, Knowledge, Explain Agents.

Coordinator điều phối workflow.

Viết test cho từng agent (tests/).

✅ Deliverable:
Hệ thống multi-agent cơ bản chạy được.

Day 7 — Demo & UI

Xây dựng Streamlit app (src/frontend/app.py).

Kết nối với FastAPI backend.

Demo input → output.

Chuẩn bị README.md + video ngắn (tuỳ chọn).

✅ Deliverable:
MVP hoàn chỉnh, có thể demo.

📦 Dependencies
langchain
openai
tiktoken
faiss-cpu
sentence-transformers
pypdf
fastapi
uvicorn
streamlit
python-dotenv
pandas
numpy
requests
pytest

🧠 Outcome After Week 1

Cấu trúc project rõ ràng, có version control.

Pipeline ingestion → embedding → RAG chạy được.

Multi-agent system hoạt động cơ bản.

Có demo UI và hướng mở rộng.

🔜 Next Step (Week 2 Preview)

Thêm Research Agent (web/API ArXiv).

Nâng cấp memory (LangGraph / Redis).

Tối ưu cost & latency (Hybrid model).

Viết benchmark test cho performance.