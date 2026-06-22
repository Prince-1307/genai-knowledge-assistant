# Personal Knowledge Assistant — Capstone

A fully deployed, end-to-end agentic AI system that lets users upload their own PDF documents and have intelligent conversations with them. Built as the capstone of a 3-project GenAI learning series, merging RAG, multi-tool agents, and conversation memory into one unified system exposed via a REST API and interactive web UI.

## 🔗 Live Demo
**Frontend:** https://genai-knowledge-assistant.streamlit.app
**API Docs:** https://knowledge-assistant-api.onrender.com/docs

## What it does
- Upload any PDF → it gets chunked, embedded, and indexed in real time
- Ask questions in natural language → the agent decides whether to search your documents, search the web, calculate, or answer from its own knowledge
- Maintains conversation memory across turns — follow-up questions like "What is his date of birth?" resolve correctly using prior context
- Full REST API with interactive documentation (FastAPI + Swagger UI)
- Clean web UI with sidebar document management and chat interface (Streamlit)

## Architecture

User (Streamlit UI)
    ↓(HTTP requests)
FastAPI Backend (Render)
    ↓
Unified Agent (ReAct loop)
    ├── search_documents → ChromaDB (hybrid vector + keyword retrieval)
    ├── web_search → Tavily API
    ├── calculate → local Python eval
    └── get_current_date → local datetime
    ↓
Gemini API (embeddings + generation)

## How it was built — project progression
This capstone merges three independently-built projects:
- **Project 1 (RAG):** PDF ingestion, chunking, embeddings, hybrid retrieval
- **Project 2 (Agent):** multi-tool ReAct agent, function calling, multi-step chaining
- **Project 3 (Memory + Eval):** conversation memory, context compression, LLM-as-judge evaluation

Each project's core logic lives in `src/core/`, unified into a single `Agent` class.

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/ask` | Send a question, get an answer |
| POST | `/upload` | Upload a PDF to the knowledge base |
| GET | `/documents` | List indexed documents |
| POST | `/reset` | Clear conversation memory |
| GET | `/history` | Get conversation history |

## Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM + Embeddings | Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`) |
| Web Search | Tavily API |
| Vector DB | Pinecone (cloud-hosted, persistent) |
| API Framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Backend Deployment | Render (free tier) |
| Frontend Deployment | Streamlit Cloud (free tier) |

## Key technical decisions & findings

**1. RAG as a tool, not a fixed first step**
Instead of always retrieving from documents before generating, the agent treats `search_documents` as one of several tools it can optionally call — alongside web search, calculator, and date. This means the agent dynamically decides when local document knowledge is relevant vs. when web search or direct reasoning is better.

**2. Hybrid retrieval (semantic + keyword)**
Pure semantic search failed on structured documents (forms, tables) — the correct chunk ranked 14th by embedding distance alone. Fixed by combining vector similarity with keyword overlap re-ranking, correctly surfacing relevant chunks that embeddings alone missed.

**3. Critical fallback-hallucination bug**
When tool-calling generation failed, a naive fallback (retry without tools) caused the model to fabricate plausible-sounding but completely wrong answers (invented names, dates) rather than admitting failure. Fixed by explicitly instructing the fallback to report a technical error honestly — a critical safety finding with real implications for production AI systems.

**4. Provider reliability varies significantly**
Groq's Llama models produced intermittent malformed JSON for tool calls. Gemini's native function calling was significantly more reliable for this use case. Real systems need to account for model-level reliability differences, not just capability differences.

**5. Cloud-hosted vector store for persistence**
Migrated from local ChromaDB to Pinecone (cloud-hosted) to solve the ephemeral filesystem problem on Render's free tier. Local ChromaDB resets on every server restart; Pinecone persists across restarts, redeploys, and scaling events. The retrieval interface stayed identical — only the storage backend changed, confirming the abstraction boundary between `search_documents()` and the underlying vector store was clean.

## Known limitations
- Free-tier Gemini API rate limits cause occasional 503 errors under heavy use
- Response latency is 3-8 seconds due to multiple sequential Gemini API calls per question — streaming responses would improve perceived performance
- Tool-call reliability on open-source models (Groq/Llama) is lower than proprietary models (Gemini) for structured function calling

## How to run locally
1. `cd capstone`
2. `python -m venv venv` then activate
3. `pip install -r requirements.txt`
4. Create `.env` with `GEMINI_API_KEY`, `TAVILY_API_KEY`, `GROQ_API_KEY`
5. `python src/core/retrieval.py` to build initial vector store (optional)
6. Terminal 1: `python -m uvicorn src.api.main:app --reload`
7. Terminal 2: `python -m streamlit run src/frontend/app.py`

## What I learned
- Merging multiple systems requires clean abstraction boundaries — keeping `core/` separate from `api/` made the FastAPI layer a thin wrapper, not a tangled mess
- File uploads through an API (`UploadFile`, `python-multipart`, async file reading) are a distinct, important pattern from local file I/O
- Deployment surfaces real constraints invisible in local development: ephemeral filesystems, missing env variables, cold starts, platform-specific behavior
- Model reliability is as important as model capability — a model that occasionally generates malformed structured output needs retry logic and safe fallbacks
- A naive fallback can be worse than no fallback if it causes confident hallucination instead of honest failure