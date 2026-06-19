import os
from dotenv import load_dotenv
from google import genai
import chromadb
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"   # fast, free-tier friendly generative model

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def keyword_overlap_score(question, chunk_text):
    """
    Counts how many meaningful words from the question appear in the chunk.
    Returns a simple overlap score.
    """
    stopwords = {"what", "is", "the", "of", "a", "an", "in", "on", "for", "to", "and", "are", "was", "were"}

    # Step 1: extract words from question, lowercase, remove punctuation
    question_words = set(re.findall(r"\w+", question.lower())) - stopwords

    # Step 2: extract words from chunk_text the same way
    chunk_words = set(re.findall(r"\w+", chunk_text.lower())) - stopwords

    # Step 3: count overlap
    overlap = len(question_words & chunk_words)
    return overlap


def retrieve_chunks(question, top_k=5, wide_k=15):
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="knowledge_base")

    query_embedding = embed_query(question)

    # Stage 1: wide vector search
    results = collection.query(query_embeddings=[query_embedding], n_results=wide_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Stage 2: re-rank using combined score (lower distance = better, higher keyword overlap = better)
    scored = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        kw_score = keyword_overlap_score(question, doc)
        # combine: normalize distance influence vs keyword boost
        # lower combined_score = better (we'll sort ascending)
        combined_score = dist - (kw_score * 0.1)   # each keyword match reduces "distance" by 0.1
        scored.append((combined_score, doc, meta))

    # Step 4: sort by combined_score ascending, take top_k
    scored.sort(key=lambda x: x[0])
    top = scored[:top_k]

    # Step 5: rebuild results in the same shape build_prompt() expects
    final_documents = [item[1] for item in top]
    final_metadatas = [item[2] for item in top]

    return {"documents": [final_documents], "metadatas": [final_metadatas]}


def embed_query(text):
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values



def build_prompt(question, results):
    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[Source: {meta['filename']}, page {meta['page_number']}]\n{doc}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have that information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask(question):
    results = retrieve_chunks(question)
    prompt = build_prompt(question, results)

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    print("\n--- Answer ---")
    print(response.text)

    print("\n--- Sources ---")
    for meta in results["metadatas"][0]:
        print(f"- {meta['filename']}, page {meta['page_number']}")

    


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        ask(question)


