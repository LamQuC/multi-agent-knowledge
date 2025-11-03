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
        # We can use simple LLM-based classifier with explain model
        self.llm = create_langchain_llm(model_name=config.MODEL_EXPLAIN, temperature=0.0)
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)

    def _extract_text_from_llm_response(self, resp) -> str:
        """Robustly extract textual content from different LLM response shapes."""
        try:
            # If ChatModel.invoke returned an object with .content
            if hasattr(resp, "content"):
                return resp.content
            # If generate-style with generations
            if hasattr(resp, "generations"):
                return resp.generations[0][0].text
            # If it's a dict with common keys
            if isinstance(resp, dict):
                return (resp.get("answer") or resp.get("result") or resp.get("output") or resp.get("output_text") or str(resp))
        except Exception:
            pass
        return str(resp)

    def classify_intent(self, query: str) -> str:
        # lightweight prompt — use LLM.invoke (safer)
        prompt = (
            "Classify the user's query into one of: retrieve, explain, code, other.\n"
            f"Return only one word.\n\nQuery: {query}"
        )

        try:
            resp = self.llm.invoke(prompt)
            text = self._extract_text_from_llm_response(resp).strip().lower()
        except Exception:
            text = ""

        # fallback heuristics if LLM failed or returned unexpected token
        if not text:
            text = query.lower()
            if "code" in text or "viết" in text:
                intent = "code"
            elif "giải thích" in text or "explain" in text:
                intent = "explain"
            elif "paper" in text or "research" in text or "tóm tắt" in text:
                intent = "retrieve"
            else:
                intent = "other"
            return intent

        # get first token/word
        intent_word = text.split()[0]
        if intent_word not in ["retrieve", "explain", "code", "other"]:
            return "other"
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

        if intent == "retrieve":
            result = self.knowledge_agent.run(user_query)
        elif intent == "explain":
            result = self.explain_agent.run(user_query)
        elif intent == "code":
            result = self.code_agent.run(user_query)
        else:
            # fallback general answer using explain model via invoke
            llm = create_langchain_llm(model_name=config.MODEL_EXPLAIN)
            try:
                resp = llm.invoke(user_query)
                text = resp.content if hasattr(resp, "content") else (
                    resp.generations[0][0].text if hasattr(resp, "generations") else str(resp)
                )
            except Exception:
                text = "Sorry, I couldn't produce an answer right now."
            result = {"answer": text}

        # normalize result
        if isinstance(result, str):
            result = {"answer": result}
        if "answer" not in result:
            result["answer"] = ""

        self.short_memory.add_message("assistant", result["answer"])

        # optional push to long-term memory handled inside agents
        return {"intent": intent, **result}
