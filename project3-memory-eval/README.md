# Memory + Evaluation System (RAG with Conversation Memory)

Extends the Project 1 RAG pipeline with conversation memory (short-term + compressed long-term) and a measurable evaluation harness using LLM-as-judge.

## What it does
- Answers questions from PDF documents (same RAG core as Project 1)
- Maintains conversation memory so follow-up questions ("What's his date of birth?") resolve correctly using prior context
- Compresses older conversation history into a summary once it exceeds a turn threshold, to manage context window size
- Evaluates system accuracy against a held-out test set using a second LLM call as an automated judge

## Architecture

Question → [check memory: compress old turns if needed] 
→ Retrieve chunks (hybrid search)
→ Build prompt (context + recent history + summary) 
→ Generate answer (Groq/Llama 3.3)
→ Save turn to memory

Eval: Test set (Q + expected answer) 
→ Run through full pipeline 
→ LLM-as-judge compares actual vs expected 
→ Accuracy report

## Multi-Provider Setup
- **Embeddings:** Gemini API (`gemini-embedding-001`)
- **Generation:** Groq API (`llama-3.3-70b-versatile`)

Switched generation to Groq after hitting Gemini's free-tier rate limits during testing — a practical, realistic constraint. This also demonstrates the system isn't tightly coupled to one provider: all generation calls route through a single `generate()` function, making provider swaps a one-function change.

## Evaluation Results
**9/10 (90%) accuracy** on a 10-question held-out test set, graded by LLM-as-judge for semantic correctness (not exact string match).

## Known Limitations (found via systematic debugging, not guessed)

**1. Summarization can silently drop facts.** Tested a 5-turn conversation where the compression summary claimed "no other details were discussed" despite two facts (DOB, disability status) having been covered earlier. The system still answered correctly because the relevant fact remained in the full-detail turn window at query time — but a longer conversation could expose this and cause wrong answers. Root cause: the summarization prompt doesn't enforce exhaustive fact retention. Fix would require a stricter summarization prompt; not implemented due to time scope.

**2. Retrieval can miss relevant chunks entirely, not just under-rank them.** The eval's one failure case ("which institution did the candidate attend?") revealed the correct chunk ranked 16th by vector distance — outside the `wide_k=15` candidate pool used for hybrid re-ranking, so the keyword-boost step never got a chance to act on it. This differs from a separate, earlier-diagnosed Project 1 issue where the correct chunk was in the candidate pool but lost on re-ranking. Confirmed via direct embedding-distance inspection (see `debug_chunks.py`). Fix would involve increasing `wide_k` or adding multi-query expansion; documented rather than fixed, given the system was already at 90% accuracy.

## Tech Stack
- **Generation:** Groq API (Llama 3.3 70B)
- **Embeddings:** Google Gemini API
- **Vector DB:** ChromaDB
- **Eval method:** LLM-as-judge (semantic correctness grading)

## How to run
1. `cd project3-memory-eval`
2. `python -m venv venv` then activate it
3. `pip install -r requirements.txt`
4. Add to `.env`: `GEMINI_API_KEY=...` and `GROQ_API_KEY=...`
5. Drop PDFs into `data/`
6. `python src/embed_and_store.py` to build the vector store
7. `python src/ask.py` for interactive Q&A with memory
8. `python src/run_eval.py` to run the evaluation harness

## What I learned
- Short-term memory (conversation buffer) and long-term memory (compressed summary) solve different problems and need different strategies
- Summarization is lossy by default — "summarize the conversation" isn't a safe operation without explicit constraints on what must be preserved
- A fixed test set + LLM-as-judge turns "does this seem to work" into a measurable, defensible claim (90% accuracy) rather than a vague impression
- Real systems hit provider rate limits; designing around a single `generate()` abstraction made switching from Gemini to Groq for generation a one-function change, not a rewrite
- Retrieval failures have more than one root cause (lost in re-ranking vs. never entering the candidate pool at all) — diagnosing precisely matters more than just "increasing top_k and hoping"

## Next steps
- Merge with Project 1 (RAG) and Project 2 (Agent) into a unified capstone system
- Improve summarization prompt to enforce exhaustive fact retention
- Expand eval set size for more statistically meaningful accuracy numbers