from src.agents.router import RouterAgent

def test_router_basic():
    router = RouterAgent()
    queries = [
        "Explain what a transformer model is.",
        "Show me Python code for attention mechanism.",
        "Summarize this paper about reinforcement learning.",
        "Hi there!"
    ]
    for q in queries:
        result = router.run(q)
        print(f"Query: {q}")
        print(f"Intent: {result['intent']}")
        print("------")

if __name__ == "__main__":
    test_router_basic()
