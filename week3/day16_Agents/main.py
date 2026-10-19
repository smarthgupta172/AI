import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from tavily import TavilyClient

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# 1. ENVIRONMENT & CLIENT INITIALIZATION
# ============================================================

# Search for .env in current or parent directories
search_dir = Path(__file__).resolve().parent
for _ in range(4):
    candidate = search_dir / ".env"
    if candidate.exists():
        load_dotenv(candidate)
        break
    search_dir = search_dir.parent
else:
    load_dotenv()

# Initialize API clients using python-dotenv loaded keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file!")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from your .env file!")

groq = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Model configuration:
# Note: "qwen/qwen3.6-27b" was requested. On Groq, the active available model in this
# family is "qwen/qwen3.8-27b". We attempt the requested model first and fall back seamlessly.
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
FALLBACK_MODEL = "qwen/qwen3.8-27b"


# ============================================================
# 2. TOOL DEFINITIONS (PYTHON FUNCTIONS)
# ============================================================

def web_search(query: str) -> str:
    """
    Perform web search using Tavily and return relevant information.
    """
    try:
        response = tavily.search(query=query, max_results=5)

        # If Tavily returned a direct answer, prioritize it
        if response.get("answer"):
            return response["answer"]

        # Otherwise extract snippets from search results
        results = response.get("results", [])
        if not results:
            return "No reliable public information could be found for this query."

        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "Untitled")
            content = res.get("content", "").strip()
            url = res.get("url", "")
            if content:
                formatted_results.append(f"[{i}] {title}\nURL: {url}\nSummary: {content}\n")

        if not formatted_results:
            return "No reliable public information could be found for this query."

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


def calculate(expression: str) -> str:
    """
    Evaluate a basic mathematical expression string and return the answer.
    Example expressions: '2*2', '1548 * 27 - 350 / 5', '2 ** 10'.
    """
    try:
        import math

        # Allow basic arithmetic and math utilities safely
        allowed_globals = {
            "__builtins__": None,
            "math": math,
            "abs": abs,
            "round": round,
            "pow": pow,
            "min": min,
            "max": max,
        }
        clean_expression = expression.strip()
        result = eval(clean_expression, allowed_globals, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


# Map tool names to callable Python functions
available_tools = {
    "web_search": web_search,
    "calculate": calculate,
}


# ============================================================
# 3. STANDARD TOOLS OBJECT FOR LLM (STANDARD FORMAT)
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web using Tavily. MUST be called whenever the user asks about ANY person "
                "(including individuals, creators, YouTubers, founders, celebrities, public figures, executives), "
                "companies, websites, products, current events, recent facts, news, real-time weather, or any topic that "
                "benefits from up-to-date web information. Never assume a person is private or answer from memory without searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string to look up on the web (e.g. person name, entity, topic, or question)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates mathematical and arithmetic expressions (e.g. '2*2', '1548 * 27 - 350 / 5', '2**8'). "
                "MUST be called whenever a mathematical calculation or arithmetic expression evaluation is requested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression string to calculate (e.g. '1548 * 27 - 350 / 5')."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# ============================================================
# 4. SYSTEM PROMPT & TOOL-SELECTION LOGIC
# ============================================================

SYSTEM_PROMPT = """You are an intelligent AI assistant equipped with two tools: `web_search` and `calculate`.

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. Web Search Policy:
   - If the user asks about ANY person (e.g. "who is [Name]", YouTubers, creators, celebrities, executives, public figures, or any named individual), company, organization, website, current event, recent fact, news, weather, or real-world information: you MUST call the `web_search` tool first.
   - Do NOT assume a person is a private individual just because the query only contains their name. Always search the web first.
   - Do NOT answer about people or current facts from your internal training memory.
   - After receiving the search results:
     * If reliable public information is found, summarize that information clearly and accurately.
     * If the search finds no useful or reliable information, clearly state that reliable public information could not be found.
2. Calculator Policy:
   - If the user asks to calculate, compute, or evaluate a mathematical expression, you MUST call the `calculate` tool.
3. Direct Answers:
   - Only answer directly without calling tools for purely conversational messages (e.g., greetings like "hello", "how are you") or questions about your own general capabilities that do not require facts or math.
4. Immediate Execution:
   - When calling a tool, do NOT output conversational preambles or thoughts. Invoke the tool immediately.
"""


def detect_preferred_tool(query: str):
    """
    Tool-selection logic to strongly prefer appropriate tools:
    - Returns 'calculate' for math expressions.
    - Returns 'web_search' for people, weather, news, current facts, etc.
    - Returns None for conversational chit-chat / general queries.
    """
    q = query.lower().strip()

    # Mathematical expression patterns
    math_patterns = [
        r"^\s*calculate\b",
        r"^\s*compute\b",
        r"^\s*evaluate\b",
        r"[\d\)]\s*[\+\-\*\/\%]\s*[\d\(]",
        r"^\s*[\d\.\(\)\+\-\*\/\^\s]+\s*$",
    ]
    if any(re.search(pat, q) for pat in math_patterns):
        return "calculate"

    # Person, entity, weather, news, current facts patterns
    search_patterns = [
        r"\bwho (is|was|are|were)\b",
        r"\bweather\b",
        r"\bcurrent\b",
        r"\blatest\b",
        r"\bnews\b",
        r"\bceo\b",
        r"\bfounder\b",
        r"\byoutuber\b",
        r"\bcreator\b",
        r"\bcelebrity\b",
        r"\bpresident\b",
        r"\bprime minister\b",
        r"\bsearch\b",
        r"\bprice of\b",
        r"\btemp(erature)?\b",
        r"\bscore\b",
    ]
    if any(re.search(pat, q) for pat in search_patterns):
        return "web_search"

    return None


# ============================================================
# 5. LLM CALL HELPER WITH MODEL FALLBACK
# ============================================================

def call_groq(messages, tools=None, tool_choice=None):
    """
    Call Groq chat completions API with automatic model fallback support.
    """
    kwargs = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 400,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            return groq.chat.completions.create(model=model_name, **kwargs)
        except BadRequestError as e:
            # Re-raise BadRequestError so tool fallback handler can inspect failed_generation
            raise e
        except Exception as e:
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg:
                continue
            raise e

    raise RuntimeError(f"Failed to call Groq API with models: {[PRIMARY_MODEL, FALLBACK_MODEL]}")


def _parse_failed_generation(failed_gen: str):
    """
    Extract function name and arguments if Qwen emits raw XML format
    and Groq returns code='tool_use_failed'.
    """
    fn_match = re.search(r'<function=([a-zA-Z0-9_]+)>', failed_gen)
    if not fn_match:
        return None, {}
    fn_name = fn_match.group(1)
    params = {}
    for pm in re.finditer(r'<parameter=([a-zA-Z0-9_]+)>\n?(.*?)\n?</parameter>', failed_gen, re.DOTALL):
        params[pm.group(1)] = pm.group(2).strip()
    return fn_name, params


# ============================================================
# 6. AGENT EXECUTION LOGIC
# ============================================================

def run_agent(user_query: str) -> str:
    """
    Runs the AI agent:
    1. Determines tool choice (uses intelligent preference or 'auto').
    2. Sends query to Groq LLM with tools.
    3. If tool is called, invokes the tool and passes results back.
    4. Produces the final answer based on the tool result.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        }
    ]

    print(f"\nUser Input: \"{user_query}\"")

    # Determine tool choice: prefer specific tool if detected, otherwise default to 'auto'
    preferred_tool = detect_preferred_tool(user_query)
    initial_tool_choice = (
        {"type": "function", "function": {"name": preferred_tool}}
        if preferred_tool
        else "auto"
    )

    try:
        # Step 1: Let the LLM decide or execute the selected tool
        response = call_groq(messages=messages, tools=tools, tool_choice=initial_tool_choice)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Fallback check: if no tool was called but preferred_tool indicates a tool was needed
        if not tool_calls and preferred_tool:
            response = call_groq(
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": preferred_tool}}
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                function_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                print(f"  [Tool Selected] -> {function_name}")
                print(f"  [Arguments]     -> {function_args}")

                # Execute the corresponding Python tool function
                if function_name in available_tools:
                    tool_fn = available_tools[function_name]
                    tool_output = tool_fn(**function_args)
                else:
                    tool_output = f"Error: Tool '{function_name}' is not recognized."

                print(f"  [Tool Result]   -> {str(tool_output)[:120]}..." if len(str(tool_output)) > 120 else f"  [Tool Result]   -> {tool_output}")

                # Append tool result back into messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(tool_output),
                })

            # Step 2: Second call to LLM with tool output for final synthesis
            second_response = call_groq(messages=messages)
            final_answer = second_response.choices[0].message.content
        else:
            # LLM answered directly without calling tools
            final_answer = response_message.content

    except BadRequestError as e:
        # Handle Qwen trailing token edge-case on Groq ('tool_use_failed')
        body = getattr(e, "body", {}) or {}
        err = body.get("error", {}) if isinstance(body, dict) else {}
        failed_gen = err.get("failed_generation", "")

        fn_name, fn_args = _parse_failed_generation(failed_gen)
        # If parsing didn't find name but preferred_tool exists, use preferred_tool
        if not fn_name and preferred_tool:
            fn_name = preferred_tool
            if fn_name == "web_search":
                fn_args = {"query": user_query}
            elif fn_name == "calculate":
                fn_args = {"expression": user_query}

        if fn_name and fn_name in available_tools:
            print(f"  [Tool Selected (Recovered)] -> {fn_name}")
            print(f"  [Arguments]                 -> {fn_args}")

            tool_fn = available_tools[fn_name]
            tool_output = tool_fn(**fn_args)

            print(f"  [Tool Result]               -> {str(tool_output)[:120]}..." if len(str(tool_output)) > 120 else f"  [Tool Result]               -> {tool_output}")

            # Prompt LLM with the tool output to generate the final answer
            recovery_messages = list(messages)
            recovery_messages.append({
                "role": "assistant",
                "content": f"I used {fn_name} with arguments {json.dumps(fn_args)}."
            })
            recovery_messages.append({
                "role": "user",
                "content": (
                    f"The tool output from {fn_name} is:\n{tool_output}\n\n"
                    "Based on this tool result, please answer my original question. "
                    "If reliable public information is found, summarize it. "
                    "If no reliable information was found, clearly state that."
                )
            })
            second_response = call_groq(messages=recovery_messages)
            final_answer = second_response.choices[0].message.content
        else:
            raise e

    # Clean up Qwen reasoning tags if present
    if final_answer and "<think>" in final_answer:
        final_answer = final_answer.split("</think>")[-1].strip()

    print(f"\nFinal Answer:\n{final_answer}")
    return final_answer


# ============================================================
# 7. MAIN DEMO & INTERACTIVE PROMPT
# ============================================================

def main():
    print("=" * 65)
    print("   AI Agent: Web Search (Tavily) & Math Calculator (Groq)")
    print("=" * 65)

    # Required Test Queries
    test_queries = [
        "who is Pratyush Narain",
        "who is the current CEO of Google",
        "calculate 1548 * 27 - 350 / 5",
        "what is the weather in Delhi today",
    ]

    print("\n--- Running Required Test Queries ---")
    for query in test_queries:
        run_agent(query)
        print("-" * 65)

    # Interactive Mode
    print("\n--- Interactive Mode (type 'exit' or 'quit' to stop) ---")
    while True:
        try:
            user_input = input("\nEnter your query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            run_agent(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent.")
            break


if __name__ == "__main__":
    main()
