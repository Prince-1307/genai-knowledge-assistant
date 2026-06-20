import chromadb, os
from google import genai
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
db = chromadb.PersistentClient(path=DB_DIR)
collection = db.get_or_create_collection("knowledge_base")

q_embed = client.models.embed_content(model="models/gemini-embedding-001", contents="In which institution did the candidate take admission?").embeddings[0].values

results = collection.query(query_embeddings=[q_embed], n_results=collection.count())
for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    marker = " <-- INSTITUTE CHUNK" if "BHAVNAGAR" in doc.upper() else ""
    print(f"{i+1}. distance={dist:.4f}{marker}")