import os
from pathlib import Path
from PyPDF2 import PdfReader

class PDFParser:
    """
    Trích xuất text từ các file PDF trong thư mục chỉ định.
    """

    def __init__(self, input_dir="data/raw", output_dir="data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdf(self, pdf_path):
        """
        Đọc và trích text từ 1 file PDF.
        """
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"[ERROR] Cannot read {pdf_path}: {e}")
        return text.strip()

    def parse_all_pdfs(self):
        """
        Lặp qua tất cả PDF trong data/raw và lưu text tương ứng.
        """
        all_texts = []
        for pdf_file in self.input_dir.glob("*.pdf"):
            print(f"Parsing {pdf_file.name} ...")
            text = self.parse_pdf(pdf_file)
            if text:
                out_path = self.output_dir / f"{pdf_file.stem}.txt"
                out_path.write_text(text, encoding="utf-8")
                all_texts.append(text)
        print(f" Parsed {len(all_texts)} files.")
        return all_texts
