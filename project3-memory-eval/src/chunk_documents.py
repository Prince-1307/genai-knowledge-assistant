from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import load_pdfs
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

def chunk_documents(documents, chunk_size=200, chunk_overlap=50):
    """
    Takes the list of {filename, page_number, text} dicts from load_pdfs,
    splits each page's text into smaller chunks, and returns a new list:
    {filename, page_number, chunk_id, text}
    """

    # Step 1: create the splitter object
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        # Step 2: split this page's text into smaller pieces
        
        page_chunks = splitter.split_text(doc["text"])  

        for chunk_text in page_chunks:
            chunks.append({
                "filename": doc["filename"],
                "page_number": doc["page_number"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    docs = load_pdfs(DATA_DIR)
    chunks = chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}")
    print(f"\nSample chunk (id={chunks[0]['chunk_id']}, file={chunks[0]['filename']}, page={chunks[0]['page_number']}):")
    print(chunks[0]['text'])

    print("\n--- Checking overlap between chunk 0 and chunk 1 ---")
    print("End of chunk 0:", chunks[0]['text'][-60:])
    print("Start of chunk 1:", chunks[1]['text'][:60])