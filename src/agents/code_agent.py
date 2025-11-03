# src/agents/code_agent.py
import io
import contextlib
from src.agents.base_agent import AgentBase
from src.agents.memory.memory_manager import MemoryManager
from src.agents.memory.long_term_memory import LongTermMemory
from src.utils.llm_manager import create_langchain_llm
from src.utils.config_loader import config


class CodeAgent(AgentBase):
    """
    CodeAgent:
    - Nhận câu hỏi / yêu cầu về code.
    - Dùng LLM (model chuyên về coding) để tạo hoặc giải thích code.
    - Có thể thực thi Python code bằng sandbox exec() thay cho PythonREPLTool.
    """

    def __init__(self):
        super().__init__("CodeAgent")
        self.short_memory = MemoryManager(max_memory=config.SHORT_MEMORY_MAX)
        self.long_memory = LongTermMemory()
        self.llm = create_langchain_llm(model_name=config.MODEL_CODE, temperature=0.3)

    def _run_python_code(self, code: str) -> str:
        """Chạy code Python trong môi trường an toàn (sandbox đơn giản)."""
        buffer = io.StringIO()
        try:
            # redirect stdout để capture kết quả print()
            with contextlib.redirect_stdout(buffer):
                # Restrict builtins to a very small safe set if desired; currently empty.
                exec(code, {"__builtins__": {}})
            output = buffer.getvalue().strip()
            return output or "(no output)"
        except Exception as e:
            return f"Error: {e}"
        finally:
            buffer.close()

    def run(self, query: str) -> dict:
        self.short_memory.add_message("user", query)

        # If user sends a fenced python block, execute it
        stripped = query.strip()
        if stripped.startswith("```python") and stripped.endswith("```"):
            # remove triple backticks and optional python tag
            code = stripped.strip("`")
            if code.startswith("python"):
                code = code[len("python"):].strip()
            result = self._run_python_code(code)
            self.short_memory.add_message("assistant", result)
            return {"answer": result, "executed": True}

        # Otherwise ask LLM to produce code/explanation via invoke
        prompt = (
            "You are a helpful Python coding assistant. Read the request and respond with helpful code or explanation.\n\n"
            f"User request:\n{query}\n\nAnswer:"
        )
        try:
            # LangChain 0.3+ only needs invoke(string)
            resp = self.llm.invoke(prompt)
            if hasattr(resp, "content"):
                answer = resp.content
            elif hasattr(resp, "generations"):
                answer = resp.generations[0][0].text
            elif isinstance(resp, dict):
                answer = resp.get("answer") or resp.get("result") or str(resp)
            else:
                answer = str(resp)
        except Exception as e:
            answer = f"Error calling LLM: {e}"

        self.short_memory.add_message("assistant", answer)

        if len(self.short_memory) >= config.LONG_MEMORY_PUSH_THRESHOLD:
            self.long_memory.add_memory(self.short_memory.get_context())
            self.short_memory.clear()

        return {"answer": answer, "executed": False}
