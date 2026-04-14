"""
intent/classifier.py
Classifies user intent from transcribed text.
"""

import json
import re

INTENT_SYSTEM_PROMPT = """
You are an intent classification engine for a voice-controlled AI agent.

CRITICAL RULES — follow exactly:
- "write_code"  -> Use this when the user wants ANY code written, OR wants to create a file
  with a code extension (.py .js .ts .java .cpp .html .css .sh etc.), OR mentions a function,
  script, class, or program. This covers "create a python file", "make a js script", etc.
- "create_file" -> ONLY use this for plain non-code files: .txt, .md, .csv, .log, or empty folders.
  NEVER use create_file together with write_code. They are mutually exclusive.
- "summarize"   -> user wants text summarized. Use ALONE.
- "general_chat"-> anything else. Use ALONE.

RULE: Never return ["create_file", "write_code"] together. If code is involved, use ONLY ["write_code"].

Extract params CAREFULLY:
  - filename: infer a meaningful name if not stated (e.g. "add_numbers.py")
  - language: EXTRACT EXACTLY what programming language user mentioned (python, javascript, typescript, java, cpp, c, csharp, go, rust, ruby, php, bash, html, css, sql, etc.). Look for language keywords in the text. Default to "python" if no language mentioned.
  - description: what the code should do
  - text_to_summarize: only for summarize intent

Respond ONLY with valid JSON. No markdown, no explanation.

Examples:
Input:  "create a python file to add two numbers"
Output: {"intents":["write_code"],"params":{"filename":"add_numbers.py","language":"python","description":"a function that adds two numbers and returns the result"}}

Input:  "write a javascript retry function"
Output: {"intents":["write_code"],"params":{"filename":"retry.js","language":"javascript","description":"a retry function with exponential backoff"}}

Input:  "create a typescript file for api calls"
Output: {"intents":["write_code"],"params":{"filename":"api_calls.ts","language":"typescript","description":"functions for making API calls"}}

Input:  "create a blank readme file"
Output: {"intents":["create_file"],"params":{"filename":"readme.txt","description":"blank readme"}}

Input:  "summarize this text and save to summary.txt"
Output: {"intents":["summarize"],"params":{"filename":"summary.txt","text_to_summarize":"this text"}}
"""


def classify(text: str, config: dict, history: list = None) -> dict:
    backend  = config.get("backend", "ollama")
    messages = _build_messages(text, history)
    if backend == "ollama":
        raw = _call_ollama(messages, config.get("model", "llama3.2"))
    elif backend == "claude":
        raw = _call_claude(messages, config.get("api_key", ""))
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
    return _parse_response(raw)


def _build_messages(text: str, history: list = None) -> list:
    msgs = []
    if history:
        for h in history[-4:]:
            msgs.append({"role": "user", "content": h["transcript"]})
            msgs.append({"role": "assistant", "content": json.dumps({
                "intents": h.get("intents", []), "params": {}
            })})
    msgs.append({"role": "user", "content": text})
    return msgs


def _call_ollama(messages: list, model: str) -> str:
    try:
        import ollama
    except ImportError:
        raise ImportError("Run: pip install ollama  |  Install Ollama: https://ollama.ai")
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": INTENT_SYSTEM_PROMPT}] + messages,
    )
    return response["message"]["content"]


def _call_claude(messages: list, api_key: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=INTENT_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def _parse_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except Exception:
                result = {"intents": ["general_chat"], "params": {}}
        else:
            result = {"intents": ["general_chat"], "params": {}}
    if not result.get("intents"):
        result["intents"] = ["general_chat"]
    if "params" not in result:
        result["params"] = {}
    return result