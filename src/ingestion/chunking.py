import re
from pathlib import Path

class TextChunker:
    """
    Chia text thành các đoạn nhỏ, giúp truy vấn chính xác hơn trong RAG.
    """

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text):
        # Loại bỏ các ký tự lạ, xuống dòng thừa
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text):
        """
        Chia text thành nhiều đoạn có độ dài tương đối đồng đều.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def chunk_folder(self, input_dir="data/processed", output_path="data/processed/chunks.json"):
        import json

        all_chunks = []
        for txt_file in Path(input_dir).glob("*.txt"):
            print(f"Chunking {txt_file.name} ...")
            text = self.clean_text(txt_file.read_text(encoding="utf-8"))
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f" Saved {len(all_chunks)} chunks to {output_path}")
        return all_chunks
