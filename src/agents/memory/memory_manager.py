# src/agents/memory/memory_manager.py

from collections import deque
from datetime import datetime
import json
from pathlib import Path

class MemoryManager:
    """
    Quản lý bộ nhớ hội thoại ngắn hạn (short-term).
    - Lưu các lượt chat gần nhất vào deque.
    - Lưu/ nạp từ disk để giữ phiên giữa các restart (file JSON).
    - Cung cấp API tiện lợi: add_message, get_context, get_history_list, clear, __len__.
    """

    def __init__(self, max_memory=5, memory_file="data/processed/conversation_history.json"):
        # self.short_term: deque chứa các entry dạng dict {role, content, timestamp}
        self.short_term = deque(maxlen=max_memory)

        # Path tới file lưu history (tạo folder nếu chưa có)
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        # Nếu đã có file lưu trước đó, load vào short_term
        if self.memory_file.exists():
            try:
                # đọc JSON -> gán vào deque với maxlen
                loaded = json.load(open(self.memory_file, encoding="utf-8"))
                self.short_term = deque(loaded, maxlen=max_memory)
            except Exception:
                # Nếu có lỗi đọc, bỏ qua để tránh crash
                pass

    def add_message(self, role, content):
        """
        Thêm một message mới vào short-term memory và lưu ra file.
        - role: "user" hoặc "assistant"
        - content: văn bản message
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        # thêm vào deque (tự động giữ maxlen)
        self.short_term.append(entry)
        # lưu ngay ra disk để bền vững
        self.save()

    def get_context(self):
        """
        Trả về context dạng string nối các lượt chat gần nhất,
        phù hợp để chèn trực tiếp vào prompt.
        """
        return "\n".join([f"{m['role']}: {m['content']}" for m in list(self.short_term)])

    def get_history_list(self):
        """
        Trả về danh sách các entry (list of dict) để dùng khi cần nén/tóm tắt.
        """
        return list(self.short_term)

    def clear(self):
        """
        Xóa short-term memory (thường dùng sau khi đẩy tóm tắt vào long-term).
        """
        self.short_term.clear()
        self.save()

    def __len__(self):
        """
        Số lượt hiện có trong short-term memory.
        """
        return len(self.short_term)

    def save(self):
        """
        Ghi short_term vào file JSON.
        """
        json.dump(list(self.short_term), open(self.memory_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
