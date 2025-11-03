from src.agents.knowledge_agent import KnowledgeAgent

def test_memory_chat():
    agent = KnowledgeAgent()

    queries = [
        "What is a transformer model?",
        "Who introduced it?",
        "And what problem did it solve?",
        "Explain again briefly based on what we discussed."
    ]

    for q in queries:
        print(f" User: {q}")
        result = agent.run(q)
        print(f" Assistant: {result['answer'][:300]}...\n")

if __name__ == "__main__":
    test_memory_chat()
# This test simulates a conversation with the KnowledgeAgent to verify its memory capabilities.