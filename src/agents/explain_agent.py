# src/agents/explain_agent.py
from src.agents.base_agent import AgentBase
from src.agents.memory.memory_manager import MemoryManager
from src.agents.memory.long_term_memory import LongTermMemory
from src.utils.llm_manager import create_langchain_llm
from src.vectordb.faiss_index import VectorDB
from src.utils.config_loader import config

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class ExplainAgent(AgentBase):
    """
    ExplainAgent:
    - Dùng RetrievalChain (LangChain mới) để lấy context từ FAISS nếu có.
    - Dùng model chuyên cho explain (config.MODEL_EXPLAIN) để trả lời.
    - Nếu FAISS chưa build sẽ fallback gọi LLM trực tiếp.
    """

    def __init__(self):
        super().__init__("ExplainAgent")

        # memory
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)
        self.long_memory = LongTermMemory()

        # vector DB
        self.vector_db = VectorDB(index_path=config.FAISS_INDEX_PATH)
        self.retriever = self.vector_db.get_retriever(k=4) if self.vector_db.vectordb else None

        # LLM cho phần giải thích
        self.llm = create_langchain_llm(model_name=config.MODEL_EXPLAIN, temperature=0.2)

        # Prompt mới theo ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_template(
            "You are an AI educator. Explain the question clearly, with examples, and keep concise.\n\n"
            "Context:\n{context}\n\nQuestion:\n{input}\nAnswer:"
        )

        # Build retrieval chain (thay thế RetrievalQA)
        if self.retriever:
            self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        else:
            self.chain = None

    def _safe_extract_answer(self, obj):
        try:
            if hasattr(obj, "content"):
                return obj.content
            if hasattr(obj, "generations"):
                return obj.generations[0][0].text
            if isinstance(obj, dict):
                return obj.get("answer") or obj.get("result") or obj.get("output") or obj.get("output_text") or str(obj)
        except Exception:
            pass
        return str(obj)

    def run(self, query: str) -> dict:
        self.short_memory.add_message("user", query)

        if not self.chain:
            # fallback: không có index
            try:
                resp = self.llm.invoke(query)
                answer = self._safe_extract_answer(resp)
            except Exception:
                try:
                    resp = self.llm.generate([{"role": "user", "content": query}])
                    answer = resp.generations[0][0].text
                except Exception as e:
                    answer = f"Error calling LLM: {e}"
            retrieved_texts = []
        else:
            try:
                result = self.chain.invoke({"input": query})
            except Exception as e:
                # fallback to direct LLM if chain fails
                try:
                    resp = self.llm.invoke(query)
                    answer = self._safe_extract_answer(resp)
                    retrieved_texts = []
                except Exception:
                    answer = f"Error running retrieval chain: {e}"
                    retrieved_texts = []
            else:
                answer = result.get("answer") or result.get("output") or result.get("result") or result.get("output_text") or ""
                if not answer:
                    answer = self._safe_extract_answer(result)
                src_docs = result.get("context") or result.get("source_documents") or []
                retrieved_texts = [d.page_content for d in src_docs if hasattr(d, "page_content")]

        self.short_memory.add_message("assistant", answer)

        if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
            self.long_memory.add_memory(self.short_memory.get_context())
            self.short_memory.clear()

        return {"answer": answer, "retrieved": retrieved_texts}
