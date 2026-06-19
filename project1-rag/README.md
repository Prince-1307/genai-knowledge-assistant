# Document Q&A Assistant (RAG)

A retrieval-augmented generation (RAG) system that answers questions from your own PDF documents, with source citations, using the Gemini API.

## What it does
- Ingests PDF documents and splits them into chunks
- Embeds chunks using Gemini's embedding model and stores them in ChromaDB
- Retrieves relevant chunks for a user's question using hybrid search (semantic similarity + keyword overlap re-ranking)
- Generates a grounded answer using Gemini, with citations to source file/page

## Architecture
PDF files → Text extraction (pypdf) → Chunking (LangChain text splitter, 200 chars + overlap)
→ Embedding (Gemini embedding-001) → ChromaDB (vector store)

Question → Embed → Wide vector search (top 15) → Keyword re-ranking → Top 5 chunks
→ Prompt with context → Gemini generation → Answer + citations

## Key technical decision: Hybrid Search
Initial pure semantic search failed on certain questions (e.g., "What is the candidate's father's name?") — the correct chunk ranked 14th out of 35 by embedding distance alone, because the document's structured label-value format (Name, Father Name, Mother Name, DOB, etc.) made multiple chunks look semantically similar.

**Fix:** implemented a two-stage retrieval — wide vector search (top 15) followed by re-ranking using keyword overlap between the question and chunk text. This boosted the correct chunk back into the top results without sacrificing performance on semantically straightforward questions.

## Tech Stack
- **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Vector DB:** ChromaDB (local, persistent)
- **Chunking:** LangChain text splitters
- **PDF parsing:** pypdf

## How to run
1. Clone the repo and `cd project1-rag`
2. `python -m venv venv` then activate it
3. `pip install -r requirements.txt`
4. Add your Gemini API key to a `.env` file: `GEMINI_API_KEY=your_key_here`
5. Drop your PDFs into `data/`
6. Run `python src/embed_and_store.py` to build the vector store
7. Run `python src/ask.py` to start asking questions

## What I learned
- How embeddings and vector similarity search work in practice, not just theory
- Chunk size directly affects retrieval quality — smaller chunks helped isolate specific facts in structured documents
- Pure semantic search has real failure modes; hybrid search (semantic + keyword) is a practical, necessary fix, not over-engineering
- Debugging retrieval requires inspecting embedding distances directly, not just trusting "it should work"

## Next steps
- Add a Streamlit UI for live demos
- Build a small formal evaluation set to quantify retrieval accuracy
- Extend into an agentic system with tool use (Project 2 of this series)