import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai
import dotenv
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")  # capstone/data
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBED_MODEL = "models/gemini-embedding-001"

dotenv.load_dotenv()
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def load_pdfs(data_dir=DATA_DIR):
    documents = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            documents.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": page.extract_text()
            })
    return documents


def chunk_documents(documents, chunk_size=200, chunk_overlap=30):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    chunk_id = 0
    for doc in documents:
        for chunk_text in splitter.split_text(doc["text"]):
            chunks.append({
                "filename": doc["filename"],
                "page_number": doc["page_number"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })
            chunk_id += 1
    return chunks


def embed_text(text, client):
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def build_vector_store():
    client = get_gemini_client()
    docs = load_pdfs()
    chunks = chunk_documents(docs)

    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="knowledge_base")

    for chunk in chunks:
        embedding = embed_text(chunk["text"], client)
        collection.add(
            ids=[str(chunk["chunk_id"])],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"filename": chunk["filename"], "page_number": chunk["page_number"]}]
        )

    print(f"Vector store built. Total items: {collection.count()}")
    return collection


def keyword_overlap_score(question, chunk_text):
    stopwords = {"what", "is", "the", "of", "a", "an", "in", "on", "for", "to", "and", "are", "was", "were"}
    question_words = set(re.findall(r"\w+", question.lower())) - stopwords
    chunk_words = set(re.findall(r"\w+", chunk_text.lower())) - stopwords
    return len(question_words & chunk_words)


def search_documents(query, top_k=5, wide_k=15):
    """
    The main function the agent will call as a 'tool'.
    Returns a formatted string of relevant chunks with source citations.
    """
    client = get_gemini_client()
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="knowledge_base")

    query_embedding = embed_text(query, client)
    results = collection.query(query_embeddings=[query_embedding], n_results=wide_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    scored = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        kw_score = keyword_overlap_score(query, doc)
        combined_score = dist - (kw_score * 0.1)
        scored.append((combined_score, doc, meta))

    scored.sort(key=lambda x: x[0])
    top = scored[:top_k]

    formatted = []
    for _, doc, meta in top:
        formatted.append(f"[Source: {meta['filename']}, page {meta['page_number']}]\n{doc}")

    return "\n\n---\n\n".join(formatted)

def add_document(file_bytes, filename):
    """
    Takes raw PDF bytes and a filename, chunks + embeds + stores in ChromaDB.
    Returns number of chunks added.
    """

    client = get_gemini_client()
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="knowledge_base")

    # Step 1: read PDF from bytes instead of file path
    reader = PdfReader(io.BytesIO(file_bytes))

    # Step 2: extract text page by page (same as before)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():  # skip empty pages
            documents.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": text
            })

    # Step 3: chunk documents (reuse existing function)
    chunks = chunk_documents(documents)

    start_id = collection.count()
    for chunk in chunks:

        complete_chunk_id = start_id + chunk["chunk_id"]
        embedding = embed_text(chunk["text"], client)
        complete_chunk_id = str(complete_chunk_id)  # Ensure it's a string for ChromaDB
        collection.add(
            ids=[complete_chunk_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"filename": chunk["filename"], "page_number": chunk["page_number"]}]
        )

    return len(chunks)


def list_documents():
    """Returns list of unique filenames currently in the vector store."""
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="knowledge_base")

    result = collection.get()  # This will return all items in the collection
    filenames = list(set(m["filename"] for m in result["metadatas"]))

    return filenames


if __name__ == "__main__":
    build_vector_store()