from src.agents.knowledge_agent import KnowledgeAgent

def test_knowledge_agent():
    agent = KnowledgeAgent()
    query = "What is the Transformer architecture?"
    result = agent.run(query)
    print("Answer:", result["answer"])
    print("Context used:", result["context_used"])

if __name__ == "__main__":
    test_knowledge_agent()
