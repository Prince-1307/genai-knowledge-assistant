import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if it doesn't exist
index_name = "knowledge-base"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  # Gemini embedding-001 outputs 768-dimensional vectors
        metric="cosine",  # cosine similarity (equivalent to what ChromaDB used)
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # free tier
    )
    print(f"Index '{index_name}' created successfully")
else:
    print(f"Index '{index_name}' already exists")

# Verify
index = pc.Index(index_name)
print(f"Index stats: {index.describe_index_stats()}")