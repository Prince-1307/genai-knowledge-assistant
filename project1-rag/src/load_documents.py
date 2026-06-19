import os
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

def load_pdfs(data_dir):
    """
    Loops through all PDF files in data_dir, extracts text page by page,
    and returns a list of dicts: {filename, page_number, text}
    """
    documents = []


    # Step 1: get list of all PDF filenames in the data folder
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)

        # Step 2: open the PDF using PdfReader
        reader = PdfReader(filepath)

        # Step 3: loop through each page and extract text
        for page_num, page in enumerate(reader.pages):
            
            text = page.extract_text() 

            documents.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": text
            })

    return documents


if __name__ == "__main__":
    docs = load_pdfs(DATA_DIR)

    # Sanity check: print summary
    print(f"\nTotal pages extracted: {len(docs)}")
    if docs:
        sample = docs[0]
        print(f"\nSample (file: {sample['filename']}, page: {sample['page_number']}):")
        print(sample['text'][:300] if sample['text'] else "No text found")