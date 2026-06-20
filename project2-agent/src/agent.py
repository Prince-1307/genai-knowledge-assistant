import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import datetime

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

GEN_MODEL = "gemini-2.5-flash"

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

# New tool implementation
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


tools = types.Tool(function_declarations=[calculator_declaration, date_declaration])
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
        else:
            tool_result = "Unknown tool"

        print(f"[Tool result: {tool_result}]")
        return tool_result
    else:
        return part.text


if __name__ == "__main__":
    while True:
        msg = input("\nAsk something (or 'quit'): ")
        if msg.lower() == "quit":
            break
        answer = run_agent(msg)
        print(f"\n--- Answer ---\n{answer}")


