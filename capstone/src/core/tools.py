import os
import datetime
from tavily import TavilyClient
from .retrieval import search_documents  # our RAG retrieval, now a tool


def calculate(expression):
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_date():
    return datetime.date.today().isoformat()


def web_search(query):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    try:
        response = tavily_client.search(query=query)
        results = response.get("results", [])
        snippets = [r.get("content", "") for r in results[:3]]
        return "\n\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


# Tool declarations — what we tell Gemini/the model about each tool
TOOL_DECLARATIONS = [
    {
        "name": "calculate",
        "description": "Evaluates a basic math expression and returns the result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A math expression like '847 * 392'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_date",
        "description": "Returns today's date in YYYY-MM-DD format",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "web_search",
        "description": "Searches the web for current information, recent events, or anything not in your training data",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_documents",
        "description": "Searches the user's personal documents (PDFs) for specific information like personal details, records, or facts contained in their files",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in the documents"}
            },
            "required": ["query"]
        }
    }
]


def dispatch_tool(name, args):
    """Given a tool name and arguments, actually execute it and return the result."""
    if name == "calculate":
        return calculate(args["expression"])
    elif name == "get_current_date":
        return get_current_date()
    elif name == "web_search":
        return web_search(args["query"])
    elif name == "search_documents":
        return search_documents(args["query"])
    else:
        return "Unknown tool"