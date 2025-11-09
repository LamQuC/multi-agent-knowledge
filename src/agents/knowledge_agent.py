# src/agents/knowledge_agent.py
import os
from dotenv import load_dotenv
load_dotenv()

# LangChain primitives
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DuckDuckGo "Run" tool (returns a single summary-like string)
from langchain_community.tools import DuckDuckGoSearchRun

# Local project modules (keep FAISS + memory + llm factory)
from src.agents.base_agent import AgentBase
from src.vectordb.faiss_index import VectorDB
from src.agents.memory.memory_manager import MemoryManager
from src.agents.memory.long_term_memory import LongTermMemory
from src.utils.llm_manager import create_langchain_llm
from src.utils.config_loader import config

# Logging
import logging
logger = logging.getLogger("KnowledgeAgent")


class KnowledgeAgent(AgentBase):
    """
    KnowledgeAgent:
    - Khi KnowledgeAgent.enable_web_search == True: ALWAYS use DuckDuckGoSearchRun to fetch web info,
      then summarize + answer via LLM. Memory (short + long) kept.
    - Khi enable_web_search == False: use FAISS retriever (if exists) + LLM + memory.
    """

    # class-level toggle (can be set from backend GlobalState)
    enable_web_search = False

    def __init__(self):
        # khởi tạo base agent (tên)
        super().__init__("KnowledgeAgent")

        # FAISS vector DB (nếu index tồn tại)
        self.vector_db = VectorDB(index_path=config.FAISS_INDEX_PATH)
        # lấy retriever nếu vectordb được khởi tạo thành công
        self.retriever = self.vector_db.get_retriever(k=4) if self.vector_db.vectordb else None

        # tạo LLM chuẩn để trả lời (dùng factory của project)
        self.llm = create_langchain_llm(model_name=config.MODEL_KNOWLEDGE, temperature=0.0)

        # memory objects: short-term (conversation buffer) + long-term (persist)
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)
        self.long_memory = LongTermMemory()

        # prompt template cho retrieval chain (nếu dùng FAISS)
        self.prompt = ChatPromptTemplate.from_template(
            "You are an expert assistant. Use the context below to answer clearly.\n\n"
            "Context:\n{context}\n\nQuestion:\n{input}\n\nAnswer:"
        )

        # nếu có retriever, build retrieval chain
        if self.retriever:
            self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        else:
            self.chain = None

        # DuckDuckGo Run tool (returns a single synthesized string)
        # note: DuckDuckGoSearchRun.invoke(query) -> string (summary-like)
        self.web_tool = DuckDuckGoSearchRun()

    # helper robust extract (LLM responses có nhiều dạng)
    def _safe_extract(self, obj):
        try:
            if hasattr(obj, "content"):
                return obj.content
            if hasattr(obj, "generations"):
                return obj.generations[0][0].text
            if isinstance(obj, dict):
                return obj.get("answer") or obj.get("result") or obj.get("output") or str(obj)
        except Exception:
            pass
        return str(obj)

    # web_search_tool: dùng DuckDuckGoSearchRun.invoke để trả về 1 chuỗi
    def web_search_tool(self, query: str) -> str:
        """
        Run DuckDuckGoSearchRun.invoke(query) and return cleaned text.
        This returns one synthesized string (ideal for direct LLM summarization).
        """
        try:
            logger.info(f"[WEB] DuckDuckGoRun searching for: {query!r}")
            raw = self.web_tool.invoke(query)  # returns a string summary-like
            if not raw:
                logger.info("[WEB] DuckDuckGo returned empty.")
                return ""
            # normalize whitespace, truncate to avoid huge context
            text = " ".join(str(raw).split())
            return text[:8000]  # keep a big slice but safe
        except Exception as e:
            logger.exception(f"[WEB] DuckDuckGo error: {e}")
            return ""

    # summary_tool: nếu cần tách block dài và tóm tắt (dùng LLM)
    def summary_tool(self, text: str) -> str:
        """
        Optionally chunk & summarize a long web string using the LLM.
        Keeps memory of chunks small and robust.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = splitter.split_text(text)
            summaries = []
            for chunk in chunks:
                prompt = f"Summarize concisely the following text to support an answer:\n\n{chunk}"
                resp = self.llm.invoke(prompt)
                summaries.append(self._safe_extract(resp))
            # join chunk summaries into one paragraph
            return " ".join(summaries).strip()
        except Exception as e:
            logger.exception(f"[SUMMARY] error: {e}")
            return text[:2000] if text else ""

    # main run pipeline
    def run(self, query: str) -> dict:
        """
        Behavior:
        - always record user message in short memory
        - if class toggle enable_web_search True -> use web_search_tool -> summary_tool -> llm to answer
        - else -> try FAISS retrieval chain -> LLM; fallback to direct LLM
        - push short memory to long memory when threshold reached
        """
        logger.info(f"[RUN] query={query!r} web_toggle={KnowledgeAgent.enable_web_search}")
        # save user message to short memory
        self.short_memory.add_message("user", query)

        # if web_search toggle is enabled -> force web path
        if KnowledgeAgent.enable_web_search:
            logger.info("[RUN] Forced web-search path (toggle ON).")
            web_text = self.web_search_tool(query)
            if web_text:
                # optional: summarize long web_text into concise summary
                summary = self.summary_tool(web_text)
                # ask LLM to produce a natural answer based on the web summary
                prompt = (
                    "You are an assistant. Use the web summary below to answer the user's question clearly and concisely.\n\n"
                    f"Web summary:\n{summary}\n\nQuestion:\n{query}\n\nAnswer:"
                )
                try:
                    resp = self.llm.invoke(prompt)
                    answer = self._safe_extract(resp)
                except Exception as e:
                    logger.exception(f"[LLM] error when answering from web summary: {e}")
                    answer = "Error: failed to produce answer from web summary."

                # store assistant reply
                self.short_memory.add_message("assistant", answer)

                # push to long term memory if threshold exceeded
                if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
                    conv = self.short_memory.get_context()
                    self.long_memory.add_memory(conv)
                    self.short_memory.clear()

                return {"answer": answer, "retrieved": [summary], "source": "web"}

            # no web result -> return informative message (but still keep memory)
            logger.info("[RUN] Web search returned empty. Returning no-results message.")
            nores = "Không tìm thấy kết quả web phù hợp."
            self.short_memory.add_message("assistant", nores)
            if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
                conv = self.short_memory.get_context()
                self.long_memory.add_memory(conv)
                self.short_memory.clear()
            return {"answer": nores, "retrieved": [], "source": "web"}

        # else: web toggle off -> use FAISS retrieval if available
        logger.info("[RUN] FAISS/Local path (toggle OFF).")
        if self.chain:
            try:
                # chain.invoke returns a dict-like object with 'answer' or 'output'
                result = self.chain.invoke({"input": query})
                # robust extract
                answer = result.get("answer") or result.get("output") or result.get("result") or ""
                # attempt to extract source documents text
                source_docs = result.get("context") or result.get("source_documents") or []
                retrieved_texts = [d.page_content for d in source_docs if hasattr(d, "page_content")]
                if answer:
                    self.short_memory.add_message("assistant", answer)
                    if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
                        conv = self.short_memory.get_context()
                        self.long_memory.add_memory(conv)
                        self.short_memory.clear()
                    return {"answer": answer, "retrieved": retrieved_texts, "source": "faiss"}
            except Exception as e:
                logger.exception(f"[FAISS] retrieval error: {e}")

        # fallback: direct LLM answer (no web)
        try:
            resp = self.llm.invoke(query)
            answer = self._safe_extract(resp)
        except Exception as e:
            logger.exception(f"[LLM] direct call failed: {e}")
            answer = "Xin lỗi, tôi không thể trả lời ngay lúc này."

        # save assistant reply and push memory if needed
        self.short_memory.add_message("assistant", answer)
        if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
            conv = self.short_memory.get_context()
            self.long_memory.add_memory(conv)
            self.short_memory.clear()

        return {"answer": answer, "retrieved": [], "source": "model"}
