from src.ingestion.pdf_parsing import PDFParser
from src.ingestion.chunking import TextChunker
from src.vectordb.faiss_index import VectorDB

parser = PDFParser()
chunker = TextChunker()
vectordb = VectorDB()

# 1️⃣ Parse PDF
texts = parser.parse_all_pdfs()

# 2️⃣ Chunking
chunks = chunker.chunk_folder()

# 3️⃣ Build FAISS index
vectordb.build_index(chunks)
