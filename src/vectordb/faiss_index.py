# src/vectordb/faiss_index.py
"""
Vector DB wrapper using LangChain FAISS vectorstore + HuggingFace embeddings.
- Build from chunks (list of texts)
- Provide as_retriever() for use with RetrievalQA
"""
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.utils.config_loader import config

class VectorDB:
    def __init__(self, index_path: str = None, embedding_name: str = None):
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.embedding_name = embedding_name or config.EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_name)
        self.vectordb = None
        self._load_if_exists()

    def _load_if_exists(self):
        idx_dir = Path(self.index_path)
        if idx_dir.exists():
            try:
                self.vectordb = FAISS.load_local(self.index_path, self.embeddings)
            except Exception:
                self.vectordb = None

    def build_index(self, chunks: list):
        """
        chunks: list of strings
        """
        docs = [Document(page_content=c) for c in chunks]
        self.vectordb = FAISS.from_documents(docs, self.embeddings)
        os.makedirs(self.index_path, exist_ok=True)
        self.vectordb.save_local(self.index_path)

    def get_retriever(self, k: int = 4):
        if not self.vectordb:
            raise ValueError("Vector DB is not built")
        return self.vectordb.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 4):
        if not self.vectordb:
            return []
        docs = self.vectordb.similarity_search(query, k=k)
        return [d.page_content for d in docs]
