import os
from rag.ingestion import download_pdfs_from_gdrive, process_pdfs

def test_ingestion():
    # Google Drive folder link
    FOLDER_URL = "https://drive.google.com/drive/folders/1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_"
    
    # Step 1: Download PDFs
    local_dir = download_pdfs_from_gdrive(FOLDER_URL)
    assert os.path.exists(local_dir), "PDF download folder not found"

    # Step 2: Process PDFs into chunks
    docs = process_pdfs(local_dir, FOLDER_URL)
    assert len(docs) > 0, "No chunks were extracted from PDFs"

    # Step 3: Validate structure of one chunk
    sample = docs[0]
    assert "id" in sample, "Missing unique chunk id"
    assert "filename" in sample, "Missing filename"
    assert "drive_url" in sample, "Missing drive url"
    assert "chunk_id" in sample, "Missing chunk id"
    assert "text" in sample and len(sample["text"]) > 0, "Empty text in chunk"

    print(f"✅ Ingestion Test Passed: {len(docs)} chunks extracted across PDFs")
    print(f"Example chunk: {sample}")

if __name__ == "__main__":
    test_ingestion()
