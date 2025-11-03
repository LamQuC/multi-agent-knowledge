# tests/test_long_term_memory.py

from src.agents.memory.memory_manager import MemoryManager
from src.agents.memory.long_term_memory import LongTermMemory

def test_long_term_flow():
    # 1) tạo short-term manager và long-term
    short = MemoryManager(max_memory=4)
    long_mem = LongTermMemory()

    # 2) giả lập một hội thoại dài (4 lượt)
    short.add_message("user", "Tell me about transformer models.")
    short.add_message("assistant", "A transformer is... (assistant short reply)")
    short.add_message("user", "Who introduced it?")
    short.add_message("assistant", "It was introduced by X...")

    # 3) Lấy nội dung để thêm vào long-term
    conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in short.get_history_list()])

    # 4) Thêm vào long-term (sẽ gọi Gemini để tóm tắt)
    long_mem.add_memory(conv_text)

    # 5) Kiểm tra retrieve
    res = long_mem.retrieve_relevant_memory("Who introduced transformer", top_k=1)
    print("Retrieved long-term summary:", res)

if __name__ == "__main__":
    test_long_term_flow()
