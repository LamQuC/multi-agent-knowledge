# src/agents/base_agent.py
"""
BaseAgent: lớp cơ sở cho tất cả agent.
- Định chuẩn method run(query) -> dict {"answer":..., ...}
- Có logger tiện lợi
"""
import logging

class AgentBase:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def run(self, query: str) -> dict:
        """
        Mỗi agent phải implement run trả dict:
        {"answer": str, "retrieved": [...], "long_contexts": [...]}
        """
        raise NotImplementedError("Agent must implement run(query)")

    def info(self, msg: str):
        self.logger.info(f"[{self.name}] {msg}")
