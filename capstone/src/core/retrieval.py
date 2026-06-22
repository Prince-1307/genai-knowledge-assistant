import os
import re
import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from google import genai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

EMBED_MODEL = "models/gemini-embedding-001"
PINECONE_INDEX = "knowledge-base"


def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(PINECONE_INDEX)


def embed_text(text, client):
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def load_pdfs(data_dir=DATA_DIR):
    documents = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "filename": filename,
                    "page_number": page_num + 1,
                    "text": text
                })
    return documents


def chunk_documents(documents, chunk_size=200, chunk_overlap=30):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
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


def keyword_overlap_score(question, chunk_text):
    stopwords = {"what", "is", "the", "of", "a", "an", "in", "on", "for",
                 "to", "and", "are", "was", "were"}
    question_words = set(re.findall(r"\w+", question.lower())) - stopwords
    chunk_words = set(re.findall(r"\w+", chunk_text.lower())) - stopwords
    return len(question_words & chunk_words)


def add_document(file_bytes, filename):
    """Ingests a PDF from bytes into Pinecone."""
    gemini_client = get_gemini_client()
    index = get_pinecone_index()

    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": text
            })

    chunks = chunk_documents(documents)

    # Get current vector count for unique ID offset
    stats = index.describe_index_stats()
    offset = stats.total_vector_count

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk["text"], gemini_client)
        vectors.append({
            "id": str(offset + i),
            "values": embedding,
            "metadata": {
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "text": chunk["text"]
            }
        })

    # Upsert in batches of 100 (Pinecone best practice)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    return len(chunks)


def list_documents():
    """Returns list of unique filenames in Pinecone."""
    index = get_pinecone_index()
    # Pinecone doesn't have a direct "list all metadata" — we use a dummy query
    results = index.query(
        vector=[0.0] * 3072,
        top_k=100,
        include_metadata=True
    )
    filenames = list(set(
        match.metadata["filename"]
        for match in results.matches
        if match.metadata
    ))
    return filenames


def search_documents(query, top_k=5, wide_k=15):
    """Hybrid search: vector similarity + keyword re-ranking."""
    gemini_client = get_gemini_client()
    index = get_pinecone_index()

    query_embedding = embed_text(query, gemini_client)

    results = index.query(
        vector=query_embedding,
        top_k=wide_k,
        include_metadata=True
    )

    scored = []
    for match in results.matches:
        text = match.metadata.get("text", "")
        filename = match.metadata.get("filename", "")
        page_number = match.metadata.get("page_number", 0)
        distance = 1 - match.score  # Pinecone returns similarity (higher=better), convert to distance
        kw_score = keyword_overlap_score(query, text)
        combined_score = distance - (kw_score * 0.1)
        scored.append((combined_score, text, filename, page_number))

    scored.sort(key=lambda x: x[0])
    top = scored[:top_k]

    formatted = []
    for _, text, filename, page_number in top:
        formatted.append(f"[Source: {filename}, page {page_number}]\n{text}")

    return "\n\n---\n\n".join(formatted)


def build_vector_store():
    """Batch ingest all PDFs from data/ folder into Pinecone."""
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    gemini_client = get_gemini_client()
    index = get_pinecone_index()

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk["text"], gemini_client)
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "text": chunk["text"]
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    print(f"Vector store built. Total vectors: {len(vectors)}")


if __name__ == "__main__":
    build_vector_store()