# Multi-Tool Agent (ReAct Pattern)

An autonomous agent built with the Gemini API that reasons about user questions, decides which tools to use, executes them, and chains multiple tool calls together until it has enough information to answer — implementing the ReAct (Reason + Act) pattern.

## What it does
- Accepts natural language questions
- Decides whether a question needs a tool (calculator, current date, or web search) or can be answered directly from the model's own knowledge
- Executes the chosen tool, observes the result, and either calls another tool or gives a final answer
- Handles multi-step questions requiring 2+ tools chained together (e.g., "what's the date 100 days from now")

## Architecture

User question
→ Gemini decides: answer directly, OR request a tool call
→ [if tool requested] your code executes the real tool (calculator / date / web search)
→ tool result fed back into the conversation
→ Gemini decides again: another tool, OR final answer
→ (loop continues until Gemini gives a final text answer, max 5 steps)

## Tools
| Tool | Purpose | Notes |
|---|---|---|
| `calculate` | Evaluates math expressions | Local, no external API |
| `get_current_date` | Returns today's date | Local, no external API |
| `web_search` | Searches the web via Tavily API | Used for current events / info outside training data |

## Key technical decisions & bugs fixed

**1. Incomplete agent loop (early bug):** Initially, when a tool was called, the raw tool output was returned directly to the user instead of being sent back to Gemini for a final synthesized answer. Fixed by completing the full ReAct cycle: tool result → fed back to model → model writes the actual answer.

**2. Single-step limitation:** The agent could only call one tool per question, so it failed on questions needing multiple chained tool calls. Fixed by converting the single-call logic into a loop that tracks full conversation history (including all tool calls and results) and continues until Gemini itself signals it's ready to answer — capped at 5 steps to avoid infinite loops.

**3. Tool-call over-restriction:** Adding tool declarations initially made the model refuse to answer general knowledge questions outside its tools' scope (e.g., "What's the capital of France?"). Fixed with an explicit system instruction clarifying tools are optional and should only be used when relevant.

## Tech Stack
- **LLM:** Google Gemini API (`gemini-2.5-flash`) with native function calling
- **Search:** Tavily API (free tier, built for LLM agents)

## How to run
1. `cd project2-agent`
2. `python -m venv venv` then activate it
3. `pip install -r requirements.txt`
4. Add to `.env`: `GEMINI_API_KEY=your_key` and `TAVILY_API_KEY=your_key`
5. Run `python src/agent.py` and ask questions interactively

## What I learned
- Function calling means the model only *requests* actions — it never executes anything itself; your code is what actually calls APIs, runs functions, and has real-world effects
- The ReAct loop (reason → act → observe → repeat) is what separates a true agent from a single API call with extra steps
- Multi-step reasoning requires explicitly looping and tracking conversation history — it doesn't happen automatically just by giving the model more tools
- Tool descriptions and system instructions materially affect model behavior, including for unrelated queries — not just "does it know how to use the tool"

## Next steps
- Add more diverse tools (e.g., file reading, database queries)
- Build a Streamlit UI for live demos
- Combine with Project 1's RAG system as part of the capstone