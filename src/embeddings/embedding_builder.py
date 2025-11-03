import numpy as np

class EmbeddingBuilder:
    def embed_text(self, text: str):
        # Giả lập vector embedding bằng cách hash chuỗi
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(768)
