# GenAI Knowledge Assistant

A progressive GenAI learning project built from scratch — starting from core fundamentals and culminating in a fully deployed, agentic AI system. Each project is independently functional and builds directly toward the capstone.

## 🔗 Live Demo
**Try it here:** https://genai-knowledge-assistant.streamlit.app
**API Docs:** https://knowledge-assistant-api.onrender.com/docs

## Project Journey

### Project 1 — RAG Document Q&A [`/project1-rag`](./project1-rag)
Built a retrieval-augmented generation pipeline from scratch: PDF ingestion, chunking, embeddings, vector search, and grounded Q&A with citations.

**Core concepts:** tokenization, embeddings, vector databases, chunking strategies, semantic search, prompt engineering, grounded generation

**Key finding:** pure semantic search failed on structured documents (correct chunk ranked 14th/35). Diagnosed via embedding distance inspection and fixed with hybrid keyword + semantic re-ranking.

---

### Project 2 — Multi-Tool Agent [`/project2-agent`](./project2-agent)
Built an autonomous agent that reasons about questions, selects from multiple tools (calculator, date, web search), and chains tool calls across multiple steps using the ReAct pattern.

**Core concepts:** function calling, tool use, ReAct reasoning loop, multi-step planning, structured output, agent orchestration

**Key finding:** tool declarations affect model behavior beyond just tool availability — overly narrow descriptions caused the model to refuse general knowledge questions entirely. Fixed with explicit system instructions.

---

### Project 3 — Memory + Evaluation [`/project3-memory-eval`](./project3-memory-eval)
Extended the RAG system with conversation memory (short-term buffer + long-term compression) and a measurable evaluation harness using LLM-as-judge.

**Core concepts:** conversation memory, context compression, summarization, LLM-as-judge evaluation, hallucination detection, multi-provider setup (Gemini embeddings + Groq generation)

**Key finding:** achieved 90% accuracy (9/10) on a held-out test set. Identified that naive summarization silently drops facts, and that a second retrieval failure mode (chunk not in candidate pool at all) is distinct from a chunk losing on re-ranking.

---

### Capstone — Unified Agentic System [`/capstone`](./capstone)
Merged all three projects into one deployed system: RAG-as-tool inside a multi-tool agent with memory, exposed via FastAPI and a Streamlit UI. Users upload their own PDFs and chat with them in real time.

**Core concepts:** system architecture, API design (FastAPI), cloud deployment (Render + Streamlit Cloud), persistent vector storage (Pinecone), file upload handling, multi-provider LLM orchestration

**Key finding:** a naive fallback when tool-calling fails can cause the model to confidently hallucinate rather than admit failure — a critical safety insight fixed by explicitly instructing honest failure reporting in the fallback path.

---

## What this series covers

| Concept | Project |
|---------|---------|
| Tokenization & embeddings | P1 |
| Vector databases & similarity search | P1 |
| Chunking strategies | P1 |
| Prompt engineering & grounded generation | P1 |
| Hybrid retrieval (semantic + keyword) | P1 |
| Function calling & tool use | P2 |
| ReAct reasoning pattern | P2 |
| Multi-step agent chaining | P2 |
| Conversation memory (short + long term) | P3 |
| Context compression & summarization | P3 |
| LLM-as-judge evaluation | P3 |
| Hallucination detection | P3 |
| REST API design (FastAPI) | Capstone |
| File upload handling | Capstone |
| Cloud deployment | Capstone |
| Persistent vector storage (Pinecone) | Capstone |
| Multi-provider LLM orchestration | Capstone |
| Safe fallback design | Capstone |

## Tech Stack
- **LLM:** Google Gemini API (`gemini-2.5-flash`)
- **Embeddings:** Google Gemini (`gemini-embedding-001`)
- **Vector DB:** Pinecone (cloud-hosted)
- **Web Search:** Tavily API
- **API Framework:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Backend Deployment:** Render
- **Frontend Deployment:** Streamlit Cloud

## How to navigate this repo
Each project folder has its own README with architecture details, key findings, known limitations, and setup instructions. Start with `project1-rag/` and follow the progression — each project builds directly on the previous one.