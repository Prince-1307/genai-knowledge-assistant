import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chunk_documents import chunk_documents
from load_documents import load_pdfs

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
DB_DIR = os.path.join(os.path.dirname(__file__), "../chroma_db")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client_genai = genai.Client(api_key=api_key)

EMBED_MODEL = "models/gemini-embedding-001"


def get_embedding(text):
    result = client_genai.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return result.embeddings[0].values


def build_vector_store():
    # Step 2: load + chunk documents (reusing our earlier pipeline)
    docs = load_pdfs(DATA_DIR)
    chunks = chunk_documents(docs)
    print(f"Embedding {len(chunks)} chunks...")

    # Step 3: set up ChromaDB persistent client + collection
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="knowledge_base")

    # Step 4: embed each chunk and add to ChromaDB
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])

        collection.add(
            ids=[str(chunk["chunk_id"])],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "filename": chunk["filename"],
                "page_number": chunk["page_number"]
            }]
        )

    print(f"Done. Total items in collection: {collection.count()}")
    return collection


if __name__ == "__main__":
    build_vector_store()