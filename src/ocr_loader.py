import os
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

DATA_PATH = "data"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def extract_text_from_pdf(pdf_path):
    docs = []
    reader = PdfReader(pdf_path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            docs.append({
                "text": clean_text(text),
                "page": i + 1,
                "source": os.path.basename(pdf_path)
            })
        else:
            images = convert_from_path(
                pdf_path,
                first_page=i+1,
                last_page=i+1,
                dpi=150,
                poppler_path=r"C:\poppler-25.12.0\Library\bin"
            )

            for img in images:
                ocr_text = pytesseract.image_to_string(img)

                docs.append({
                    "text": clean_text(ocr_text),
                    "page": i + 1,
                    "source": os.path.basename(pdf_path)
                })

    return docs


def load_all_documents():
    all_docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"Processing: {file}")
            path = os.path.join(DATA_PATH, file)
            docs = extract_text_from_pdf(path)
            all_docs.extend(docs)

    return all_docs