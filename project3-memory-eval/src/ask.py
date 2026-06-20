import os
import re
from dotenv import load_dotenv
from google import genai
from groq import Groq
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "llama-3.3-70b-versatile"

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_FULL_TURNS = 3
conversation_history = []
conversation_summary = ""


def generate(prompt):
    """Single place that calls Groq for text generation."""
    response = groq_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def embed_query(text):
    result = gemini_client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def keyword_overlap_score(question, chunk_text):
    stopwords = {"what", "is", "the", "of", "a", "an", "in", "on", "for", "to", "and", "are", "was", "were"}
    question_words = set(re.findall(r"\w+", question.lower())) - stopwords
    chunk_words = set(re.findall(r"\w+", chunk_text.lower())) - stopwords
    return len(question_words & chunk_words)


def retrieve_chunks(question, top_k=5, wide_k=15):
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="knowledge_base")

    query_embedding = embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=wide_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    scored = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        kw_score = keyword_overlap_score(question, doc)
        combined_score = dist - (kw_score * 0.1)
        scored.append((combined_score, doc, meta))

    scored.sort(key=lambda x: x[0])
    top = scored[:top_k]

    final_documents = [item[1] for item in top]
    final_metadatas = [item[2] for item in top]
    return {"documents": [final_documents], "metadatas": [final_metadatas]}


def summarize_history(history):
    history_text = ""
    for turn in history:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    prompt = f"""Summarize the following conversation concisely, preserving key facts and names mentioned, in 2-3 sentences:

{history_text}

Summary:"""
    return generate(prompt)


def manage_memory():
    global conversation_summary, conversation_history

    if len(conversation_history) > MAX_FULL_TURNS:
        old_turns = conversation_history[:-MAX_FULL_TURNS]
        new_summary = summarize_history(old_turns)
        conversation_summary = (conversation_summary + " " + new_summary).strip()
        conversation_history = conversation_history[-MAX_FULL_TURNS:]

        print(f"\n[Memory compressed. Summary so far: {conversation_summary}]")
        print(f"[Keeping {len(conversation_history)} full turns in memory]\n")


def build_prompt(question, results, history):
    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[Source: {meta['filename']}, page {meta['page_number']}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    history_text = ""
    for turn in history:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    full_history_context = f"Earlier conversation summary: {conversation_summary}\n\n{history_text}" if conversation_summary else history_text

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information in the provided documents."
Use the conversation history to resolve references like "his", "that", "it", etc.

Conversation history:
{full_history_context}

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask(question):
    manage_memory()

    results = retrieve_chunks(question)
    prompt = build_prompt(question, results, conversation_history)
    answer = generate(prompt)

    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources ---")
    for meta in results["metadatas"][0]:
        print(f"- {meta['filename']}, page {meta['page_number']}")

    conversation_history.append({"question": question, "answer": answer})
    return answer


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        ask(question)