# src/agents/router.py
from src.agents.base_agent import AgentBase
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.explain_agent import ExplainAgent
from src.agents.code_agent import CodeAgent
from src.agents.memory.memory_manager import MemoryManager
from src.utils.config_loader import config
from src.utils.llm_manager import create_langchain_llm


class RouterAgent(AgentBase):
    def __init__(self):
        super().__init__("RouterAgent")
        self.llm = create_langchain_llm(model_name=config.MODEL_EXPLAIN, temperature=0.0)
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)

    def _extract_text_from_llm_response(self, resp) -> str:
        """Robustly extract textual content from different LLM response shapes."""
        try:
            if hasattr(resp, "content"):
                return resp.content
            if hasattr(resp, "generations"):
                return resp.generations[0][0].text
            if isinstance(resp, dict):
                return (
                    resp.get("answer")
                    or resp.get("result")
                    or resp.get("output")
                    or resp.get("output_text")
                    or str(resp)
                )
        except Exception:
            pass
        return str(resp)

    def classify_intent(self, query: str) -> str:
        """Use LLM + heuristics to classify query into retrieve / explain / code / other"""
        prompt = (
            "Classify the user's query into one of: retrieve, explain, code, other.\n"
            "Return only one word.\n\nQuery: {query}"
        ).format(query=query)

        try:
            resp = self.llm.invoke(prompt)
            text = self._extract_text_from_llm_response(resp).strip().lower()
        except Exception:
            text = ""

        # --- Heuristic fallback ---
        if not text:
            t = query.lower()
            if "code" in t or "viết" in t:
                return "code"
            elif "giải thích" in t or "explain" in t:
                return "explain"
            elif any(k in t for k in ["paper", "research", "tóm tắt", "thông tin", "mới nhất", "hôm nay", "tin tức"]):
                return "retrieve"
            else:
                return "retrieve"  # fallback: default to knowledge

        intent_word = text.split()[0]
        if intent_word not in ["retrieve", "explain", "code", "other"]:
            return "retrieve"
        return intent_word


class IntentRouter:
    def __init__(self):
        self.router_agent = RouterAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.explain_agent = ExplainAgent()
        self.code_agent = CodeAgent()
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)

    def route(self, user_query: str) -> dict:
        intent = self.router_agent.classify_intent(user_query)
        self.short_memory.add_message("user", user_query)

        print(f"[ROUTER] 🚦 Intent detected: {intent}")

        # --- Route theo intent ---
        if intent == "retrieve":
            result = self.knowledge_agent.run(user_query)
        elif intent == "explain":
            result = self.explain_agent.run(user_query)
        elif intent == "code":
            result = self.code_agent.run(user_query)
        else:
            # ✅ Sửa: fallback về KnowledgeAgent thay vì model thô
            print("[ROUTER] ℹ️ Fallback to KnowledgeAgent for 'other' intent")
            result = self.knowledge_agent.run(user_query)

        # --- Normalize output ---
        if isinstance(result, str):
            result = {"answer": result}
        if "answer" not in result:
            result["answer"] = ""

        self.short_memory.add_message("assistant", result["answer"])
        return {"intent": intent, **result}
