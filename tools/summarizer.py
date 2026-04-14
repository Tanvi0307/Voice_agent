"""
tools/summarizer.py
Summarizes text using an LLM.
Optionally saves the summary to output/.
"""

from tools.file_ops import create_file

SUMMARIZE_SYSTEM = """
You are a concise summarization assistant.
Summarize the provided text clearly and briefly.
Use bullet points where appropriate.
Return only the summary, no preamble.
"""


def summarize(text: str, config: dict, save_to: str = None) -> dict:
    """
    Summarize text using an LLM.

    Args:
        text:    Text to summarize
        config:  LLM config dict
        save_to: Optional filename to save summary inside output/

    Returns:
        dict with summary, file_path (if saved)
    """
    summary = _call_llm(text, config)

    result = {"success": True, "summary": summary}

    if save_to:
        file_result = create_file(save_to, summary)
        result["file_path"] = file_result["file_path"]
        result["relative_path"] = file_result["relative_path"]

    return result


# ── LLM Callers ────────────────────────────────────────────────────────────────
def _call_llm(text: str, config: dict) -> str:
    backend = config.get("backend", "ollama")

    if backend == "ollama":
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")
        resp = ollama.chat(
            model=config.get("model", "llama3.2"),
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": f"Summarize this:\n\n{text}"},
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
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SUMMARIZE_SYSTEM,
            messages=[{"role": "user", "content": f"Summarize this:\n\n{text}"}],
        )
        return resp.content[0].text

    raise ValueError(f"Unknown backend: {backend}")
