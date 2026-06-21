import os
from groq import Groq
from .tools import TOOL_DECLARATIONS, dispatch_tool
import json

GEN_MODEL = "llama-3.3-70b-versatile"
MAX_FULL_TURNS = 3
MAX_AGENT_STEPS = 5


def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate(prompt, tools=None):
    """
    Single place that calls Groq. If tools are provided, the model
    can request a tool call instead of answering directly.
    """
    client = get_groq_client()
    kwargs = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


class Agent:
    """
    A single conversation session: holds memory, runs the agent loop,
    can be reused across multiple questions.
    """

    def __init__(self):
        self.conversation_history = []
        self.conversation_summary = ""

    def _manage_memory(self):
        if len(self.conversation_history) > MAX_FULL_TURNS:
            old_turns = self.conversation_history[:-MAX_FULL_TURNS]
            history_text = "".join(f"Q: {t['question']}\nA: {t['answer']}\n\n" for t in old_turns)

            summary_prompt = f"Summarize this conversation concisely, preserving key facts and names:\n\n{history_text}\n\nSummary:"
            new_summary = generate(summary_prompt).content

            self.conversation_summary = (self.conversation_summary + " " + new_summary).strip()
            self.conversation_history = self.conversation_history[-MAX_FULL_TURNS:]

    def ask(self, question):
        self._manage_memory()

        history_text = "".join(f"Q: {t['question']}\nA: {t['answer']}\n\n" for t in self.conversation_history)
        memory_context = f"Earlier summary: {self.conversation_summary}\n\n{history_text}" if self.conversation_summary else history_text

        groq_tools = [{"type": "function", "function": decl} for decl in TOOL_DECLARATIONS]

        prompt = f"""You are a helpful assistant with access to these tools:
- search_documents: searches the user's uploaded PDF documents for any factual information (names, dates, personal records, etc.)
- calculate: for math
- get_current_date: for today's date
- web_search: for current events/info not in your training data

IMPORTANT: If the question could be about information in the user's documents (e.g. asking about "the candidate", names, personal details, records), you MUST call search_documents first before saying you don't have the information. Do not assume the conversation has no prior context — always try search_documents for factual lookups.
Conversation history:

{memory_context}

Question: {question}"""

        # Agent loop: keep calling tools until model gives a final text answer
        messages = [{"role": "user", "content": prompt}]

        for step in range(MAX_AGENT_STEPS):
            client = get_groq_client()

            response = None
            last_error = None
            for attempt in range(3):  # try up to 3 times
                try:
                    response = client.chat.completions.create(
                        model=GEN_MODEL,
                        messages=messages,
                        tools=groq_tools,
                        tool_choice="auto"
                    )
                    break  # success, stop retrying
                except Exception as e:
                    last_error = e
                    print(f"[Attempt {attempt+1} failed: {e}]")

            if response is None:
                # all retries failed — fail safely, no fabrication
                print(f"[All retries failed. Falling back to honest failure message.]")
                fallback_prompt = f"""You are a helpful assistant. You attempted to look up information but the lookup failed due to a technical error after multiple attempts.
        Tell the user honestly that you couldn't retrieve the information due to a technical issue, and do NOT make up or guess any information.

        Original question: {question}"""
                fallback_response = client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=[{"role": "user", "content": fallback_prompt}]
                )
                answer = fallback_response.choices[0].message.content
                self.conversation_history.append({"question": question, "answer": answer})
                return answer

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = dispatch_tool(tool_call.function.name, args)
                    print(f"[Step {step+1}: {tool_call.function.name}({args}) -> {str(result)[:100]}...]")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                answer = message.content
                self.conversation_history.append({"question": question, "answer": answer})
                return answer

        return "Reached max steps without a final answer."