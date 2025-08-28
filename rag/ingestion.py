import os
import hashlib
import gdown
from PyPDF2 import PdfReader
import dotenv

dotenv.load_dotenv()
FOLDER_URL = os.getenv("GOOGLE_DRIVE_FOLDER_URL")

def download_pdfs_from_gdrive(folder_url: str, output_dir: str = "data/pdfs") -> str:
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"📂 PDFs already available in {output_dir}, skipping download.")
        return output_dir
    else:
        import gdown
        gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
        return output_dir


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    Splits text into overlapping chunks (~300 tokens).
    Uses simple word splitting for now; can be upgraded to tokenizer-based.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def process_pdfs(folder_path: str, drive_url: str):
    """
    Loads all PDFs in a folder, extracts text, splits into chunks,
    and attaches metadata.
    """
    documents = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pdf"):
            fpath = os.path.join(folder_path, fname)
            raw_text = extract_text_from_pdf(fpath)
            chunks = chunk_text(raw_text)

            for idx, chunk in enumerate(chunks):
                doc_id = hashlib.md5((fname + str(idx)).encode()).hexdigest()
                documents.append({
                    "id": doc_id,
                    "filename": fname,
                    "drive_url": drive_url,
                    "chunk_id": idx,
                    "text": chunk
                })
    return documents


if __name__ == "__main__":
    # Example usage
    local_dir = download_pdfs_from_gdrive(FOLDER_URL)
    docs = process_pdfs(local_dir, FOLDER_URL)
    print(f"Extracted {len(docs)} chunks from {len(os.listdir(local_dir))} PDFs")
