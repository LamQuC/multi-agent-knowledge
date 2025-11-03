# src/agents/memory/long_term_memory.py

import os
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import generativeai as genai

# load .env để lấy GEMINI_API_KEY
load_dotenv()
# cấu hình Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class LongTermMemory:
    """
    Lưu và truy xuất ký ức dài hạn:
      - Tóm tắt conversation text bằng Gemini (gemini-2.5-flash)
      - Sinh embedding cho summary bằng SentenceTransformer
      - Thêm vào FAISS index và lưu metadata
    """

    def __init__(self,
                 memory_index_path="data/processed/memory_index.faiss",
                 meta_path="data/processed/memory_meta.npy",
                 embed_model_name="all-MiniLM-L6-v2"):
        # Đường dẫn lưu index FAISS
        self.memory_index_path = memory_index_path
        # Đường dẫn lưu metadata (list of dict)
        self.meta_path = meta_path

        # Model embedder (SentenceTransformer) -> trả về dim = 384
        self.embedder = SentenceTransformer(embed_model_name)
        self.dimension = 384  # fixed for all-MiniLM-L6-v2

        # Khởi tạo FAISS index dạng L2 flat
        self.index = faiss.IndexFlatL2(self.dimension)

        # Danh sách metadata (mỗi phần tử: {"timestamp":..., "summary":...})
        self.memory_texts = []

        # Nếu index/meta đã tồn tại, load chúng lên
        if os.path.exists(self.memory_index_path) and os.path.exists(self.meta_path):
            self._load_memory()

    def _save_memory(self):
        """
        Lưu index và metadata ra đĩa.
        - index: .faiss
        - metadata: numpy file chứa list -> dễ load lại
        """
        faiss.write_index(self.index, self.memory_index_path)
        np.save(self.meta_path, np.array(self.memory_texts, dtype=object), allow_pickle=True)

    def _load_memory(self):
        """
        Load index và metadata từ đĩa nếu có.
        """
        self.index = faiss.read_index(self.memory_index_path)
        self.memory_texts = np.load(self.meta_path, allow_pickle=True).tolist()

    def summarize_conversation(self, conversation_text: str) -> str:
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"Summarize the following conversation briefly, preserving main points and decisions:\n\n{conversation_text}"
            response = model.generate_content(prompt)
            return response.text.strip() if response.text else conversation_text[:200]
        except Exception as e:
            print(f"[LongTermMemory] Summarization error: {e}")
            # fallback để tránh crash khi API lỗi
            return conversation_text[:200]

    def add_memory(self, conversation_text: str):
        """
        Thêm 1 memory long-term:
        - Tóm tắt conversation_text
        - Tạo embedding
        - Thêm vào FAISS + metadata
        """
        # tóm tắt
        summary = self.summarize_conversation(conversation_text)
        # tạo embedding (mảng shape (1, dim))
        emb = self.embedder.encode([summary]).astype(np.float32)
        # add embedding vào faiss index
        self.index.add(emb)
        # thêm metadata
        self.memory_texts.append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        })
        # lưu state xuống đĩa
        self._save_memory()

    def retrieve_relevant_memory(self, query: str, top_k: int = 3) -> list:
        """
        Với 1 query, trả về top_k summary liên quan nhất từ long-term memory.
        """
        # nếu chưa có memory thì trả rỗng
        if len(self.memory_texts) == 0:
            return []

        # tạo embedding cho query
        q_emb = self.embedder.encode([query]).astype(np.float32)
        # search faiss index
        D, I = self.index.search(q_emb, top_k)
        # map indices sang summaries (bảo đảm i < len)
        results = []
        for i in I[0]:
            if i < len(self.memory_texts):
                results.append(self.memory_texts[i]["summary"])
        return results
