# src/agents/knowledge_agent.py
import os
from dotenv import load_dotenv
load_dotenv()

from src.agents.base_agent import AgentBase
from src.vectordb.faiss_index import VectorDB
from src.agents.memory.memory_manager import MemoryManager
from src.agents.memory.long_term_memory import LongTermMemory
from src.utils.llm_manager import create_langchain_llm
from src.utils.config_loader import config

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class KnowledgeAgent(AgentBase):
    """
    KnowledgeAgent sử dụng LangChain RetrievalChain (thay cho RetrievalQA):
    - retriever = VectorDB.get_retriever()
    - llm = Gemini model specialized for knowledge (config.MODEL_KNOWLEDGE)
    """

    def __init__(self):
        super().__init__("KnowledgeAgent")
        self.vector_db = VectorDB(index_path=config.FAISS_INDEX_PATH)
        self.retriever = self.vector_db.get_retriever(k=4) if self.vector_db.vectordb else None

        # LLM for retrieval answers
        self.llm = create_langchain_llm(model_name=config.MODEL_KNOWLEDGE, temperature=0.0)

        # memory
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)
        self.long_memory = LongTermMemory()

        # Prompt now uses ChatPromptTemplate (required by new API)
        self.prompt = ChatPromptTemplate.from_template(
            "You are an expert assistant answering technical questions using the provided context.\n"
            "If the context doesn't contain the answer, be honest.\n\n"
            "Context:\n{context}\n\nQuestion:\n{input}\nAnswer:"
        )

        # Build retrieval chain (new API)
        if self.retriever:
            self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        else:
            self.chain = None

    def _safe_extract_answer(self, obj):
        """Extract text content from different possible return shapes."""
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
        # Save user query
        self.short_memory.add_message("user", query)

        if not self.chain:
            # fallback: no index built -> call LLM directly (invoke)
            try:
                resp = self.llm.invoke(query)
                answer = self._safe_extract_answer(resp)
            except Exception:
                # best-effort fallback to .generate(...) shape
                try:
                    resp = self.llm.generate([{"role": "user", "content": query}])
                    answer = resp.generations[0][0].text
                except Exception as e:
                    answer = f"Error calling LLM: {e}"
            self.short_memory.add_message("assistant", answer)
            return {"answer": answer, "retrieved": [], "long_contexts": []}

        # Run retrieval chain
        try:
            result = self.chain.invoke({"input": query})
        except Exception as e:
            # fallback to direct LLM if chain fails
            try:
                resp = self.llm.invoke(query)
                answer = self._safe_extract_answer(resp)
            except Exception:
                answer = f"Error running retrieval chain: {e}"
            self.short_memory.add_message("assistant", answer)
            return {"answer": answer, "retrieved": [], "long_contexts": []}

        # Extract answer robustly
        answer = result.get("answer") or result.get("output") or result.get("result") or result.get("output_text") or ""
        if not answer:
            # try to pull from whatever the chain returned
            answer = self._safe_extract_answer(result)

        # source docs may be under "context" or "source_documents"
        source_docs = result.get("context") or result.get("source_documents") or []
        retrieved_texts = [d.page_content for d in source_docs if hasattr(d, "page_content")]

        # Save assistant reply
        self.short_memory.add_message("assistant", answer)

        # Push to long-term if needed
        if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
            conv = self.short_memory.get_context()
            self.long_memory.add_memory(conv)
            self.short_memory.clear()

        return {"answer": answer, "retrieved": retrieved_texts, "long_contexts": []}
