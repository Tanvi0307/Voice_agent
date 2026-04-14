"""
tools/code_gen.py
Generates code using an LLM and saves it to output/.
"""

import re
from pathlib import Path
from tools.file_ops import create_file

CODE_GEN_SYSTEM = """
You are an expert software engineer.
Generate clean, well-commented, production-quality code.
Return ONLY the raw code with no explanation, no markdown fences, no preamble.
The first line should be a comment describing what the file does.
"""


def generate_and_save(description: str, filename: str, language: str, config: dict) -> dict:
    """
    Generate code via LLM and write it to output/<filename>.

    Args:
        description: What the code should do (from user intent)
        filename:    Target filename
        language:    Programming language
        config:      LLM config dict (same as intent/classifier)

    Returns:
        dict with success, file_path, content, language
    """
    prompt = f"Write {language} code for: {description}"
    code = _call_llm(prompt, config)

    # Clean up any accidental fences
    code = _strip_fences(code)

    result = create_file(filename, code)
    result["language"] = language
    result["content"] = code
    return result


# ── LLM Callers ────────────────────────────────────────────────────────────────
def _call_llm(prompt: str, config: dict) -> str:
    backend = config.get("backend", "ollama")

    if backend == "ollama":
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")
        resp = ollama.chat(
            model=config.get("model", "llama3.2"),
            messages=[
                {"role": "system", "content": CODE_GEN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return resp["message"]["content"]

    elif backend == "claude":
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        client = anthropic.Anthropic(api_key=config["api_key"])
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=CODE_GEN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    raise ValueError(f"Unknown backend: {backend}")


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` style fences."""
    return re.sub(r"```[\w]*\n?|```", "", code).strip()
