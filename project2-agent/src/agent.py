import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

import datetime

from tavily import TavilyClient

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

GEN_MODEL = "gemini-2.5-flash"

# Tool for basic math calculations
# Step 1: define the tool's real implementation (what actually runs)
def calculate(expression):
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Step 2: define the tool's declaration (what we tell Gemini about it)
calculator_declaration = {
    "name": "calculate",
    "description": "Evaluates a basic math expression and returns the result",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression like '847 * 392'"
            }
        },
        "required": ["expression"]
    }
}

# Tool for getting the current date
def get_current_date():
    return datetime.date.today().isoformat()

# New tool declaration
date_declaration = {
    "name": "get_current_date",
    "description": "Returns today's date in YYYY-MM-DD format",
    "parameters": {
        "type": "object",
        "properties": {},  # no arguments needed
    }
}


# Tool for web search using Tavily API
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

# Step 1: tool implementation
def web_search(query):
    
    response = tavily_client.search(query=query)

    # Extract just the relevant text snippets to keep it concise
    results = response.get("results", [])
    snippets = [r.get("content", "") for r in results[:3]]  # top 3 results
    return "\n\n".join(snippets)


# Step 2: tool declaration
search_declaration = {
    "name": "web_search",
    "description": "Searches the web for current information, recent events, or anything not in your training data",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
}


tools = types.Tool(function_declarations=[calculator_declaration, date_declaration, search_declaration])

config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction="You have access to tools for calculations and getting the current date. Use them only when the question specifically requires them. For general knowledge questions, answer directly using what you know."
)



def run_agent(user_message):
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=user_message,
        config=config
    )

    part = response.candidates[0].content.parts[0]

    function_call = part.function_call

    if function_call:
        print(f"[Agent wants to call: {function_call.name}({dict(function_call.args)})]")

        if function_call.name == "calculate":
            tool_result = calculate(function_call.args["expression"])
        elif function_call.name == "get_current_date":
            tool_result = get_current_date()
        elif function_call.name == "web_search":
            tool_result = web_search(function_call.args["query"])
        else:
            tool_result = "Unknown tool"

        print(f"[Tool result: {tool_result[:200]}...]")

        follow_up = client.models.generate_content(
            model=GEN_MODEL,
            contents=f"Question: {user_message}\n\nTool result: {tool_result}\n\nBased on this tool result, answer the original question clearly and concisely.",
            config=config
        )

        return follow_up.candidates[0].content.parts[0].text
    else:
        return part.text


if __name__ == "__main__":
    while True:
        msg = input("\nAsk something (or 'quit'): ")
        if msg.lower() == "quit":
            break
        answer = run_agent(msg)
        print(f"\n--- Answer ---\n{answer}")




