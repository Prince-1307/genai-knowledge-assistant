import os
from google import genai
from google.genai import types
from .tools import TOOL_DECLARATIONS, dispatch_tool
import json

GEN_MODEL = "gemini-2.5-flash"
MAX_FULL_TURNS = 3
MAX_AGENT_STEPS = 5


def get_gemini_gen_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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

        # Build Gemini-style tools
        gemini_tools = types.Tool(function_declarations=TOOL_DECLARATIONS)
        config = types.GenerateContentConfig(
            tools=[gemini_tools],
            system_instruction=f"""You are a helpful assistant with access to these tools:
    - search_documents: searches the user's uploaded PDF documents for any factual information
    - calculate: for math expressions
    - get_current_date: for today's date
    - web_search: for current events or info not in your training data

    IMPORTANT: If the question could be about information in the user's documents (names, dates, personal details, records), you MUST call search_documents first before saying you don't have the information.

    Conversation history:
    {memory_context}"""
        )

        client = get_gemini_gen_client()
        messages = [{"role": "user", "parts": [{"text": question}]}]

        for step in range(MAX_AGENT_STEPS):
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=messages,
                config=config
            )

            part = response.candidates[0].content.parts[0]
            function_call = part.function_call

            if function_call:
                import json
                args = dict(function_call.args)
                result = dispatch_tool(function_call.name, args)
                print(f"[Step {step+1}: {function_call.name}({args}) -> {str(result)[:100]}...]")

                messages.append({"role": "model", "parts": [{"function_call": function_call}]})
                messages.append({
                    "role": "user",
                    "parts": [{"function_response": {
                        "name": function_call.name,
                        "response": {"result": result}
                    }}]
                })
            else:
                answer = part.text
                self.conversation_history.append({"question": question, "answer": answer})
                return answer

        return "Reached max steps without a final answer."