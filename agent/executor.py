"""
agent/executor.py
Orchestrates: audio -> STT -> intent -> tool execution -> result
"""

import re
from pathlib import Path

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash",
    ".html", ".css", ".sql", ".r", ".m", ".scala",
}


class AgentExecutor:
    def __init__(self):
        self.session_history = []

    def transcribe(self, audio_path: str, stt_config: dict) -> str:
        from stt.transcriber import transcribe
        return transcribe(audio_path, stt_config)

    def detect_intent(self, text: str, llm_config: dict, history: list = None) -> dict:
        from intent.classifier import classify
        return classify(text, llm_config, history or self.session_history)

    def execute(self, text: str, intent_result: dict, llm_config: dict, use_memory: bool = True) -> dict:
        intents = intent_result.get("intents", ["general_chat"])
        params  = intent_result.get("params", {})

        # If write_code and create_file both appear, drop create_file — it's redundant
        if "write_code" in intents and "create_file" in intents:
            intents = [i for i in intents if i != "create_file"]

        results = [self._dispatch(intent, text, params, llm_config) for intent in intents]
        merged  = self._merge(results)
        if use_memory:
            self.session_history.append({
                "transcript":    text,
                "intents":       intents,
                "params":        params,
                "result":        merged,
                "chat_response": merged.get("chat_response", ""),
            })
        return merged

    def _dispatch(self, intent: str, text: str, params: dict, config: dict) -> dict:
        filename = params.get("filename", "")
        if intent == "create_file" and Path(filename).suffix.lower() in CODE_EXTENSIONS:
            intent = "write_code"
        if intent == "write_code":
            return self._handle_write_code(text, params, config)
        elif intent == "create_file":
            return self._handle_create_file(params)
        elif intent == "summarize":
            return self._handle_summarize(text, params, config)
        else:
            return self._handle_chat(text, config)

    def _handle_write_code(self, text: str, params: dict, config: dict) -> dict:
        from tools.code_gen import generate_and_save
        language    = params.get("language") or _detect_language(text)
        filename    = params.get("filename") or _infer_filename(text, language)
        description = params.get("description") or text
        result = generate_and_save(description, filename, language, config)
        result["action_description"] = f"Generated {language} code -> saved to {result['relative_path']}"
        result["summary"]            = f"Code written to `{result['relative_path']}`"
        return result

    def _handle_create_file(self, params: dict) -> dict:
        from tools.file_ops import create_file, create_folder
        filename = params.get("filename") or "untitled.txt"
        if "." not in Path(filename).name:
            result = create_folder(filename)
            result["action_description"] = f"Created folder: output/{filename}"
            result["summary"]            = f"Folder `output/{filename}` created"
        else:
            result = create_file(filename)
            result["action_description"] = f"Created blank file: output/{filename}"
            result["summary"]            = f"File `output/{filename}` created"
        return result

    def _handle_summarize(self, text: str, params: dict, config: dict) -> dict:
        from tools.summarizer import summarize
        body    = params.get("text_to_summarize") or text
        save_to = params.get("filename")
        result  = summarize(body, config, save_to=save_to)
        result["action_description"] = "Summarized text" + (f" -> saved to output/{save_to}" if save_to else "")
        result["summary"] = result.get("summary", "")
        result["content"] = result.get("summary", "")
        return result

    def _handle_chat(self, text: str, config: dict) -> dict:
        context_msgs = []
        for h in self.session_history[-3:]:
            context_msgs.append({"role": "user", "content": h["transcript"]})
            if h.get("chat_response"):
                context_msgs.append({"role": "assistant", "content": h["chat_response"]})
        context_msgs.append({"role": "user", "content": text})
        response = _llm_call("You are a helpful assistant.", context_msgs, config)
        return {
            "success": True,
            "action_description": "General conversation",
            "summary": "Responded to query",
            "chat_response": response,
            "content": response,
        }

    def _merge(self, results: list) -> dict:
        if len(results) == 1:
            return results[0]
        merged = {
            "success":            all(r.get("success", True) for r in results),
            "action_description": " + ".join(r.get("action_description", "") for r in results),
            "summary":            "\n".join(r.get("summary", "") for r in results),
            "content":            "\n\n---\n\n".join(r.get("content", "") for r in results if r.get("content")),
        }
        for r in reversed(results):
            if r.get("file_path"):
                merged["file_path"]     = r["file_path"]
                merged["relative_path"] = r.get("relative_path", "")
                merged["language"]      = r.get("language", "text")
                break
        return merged


def _llm_call(system: str, messages: list, config: dict, max_tokens: int = 1024) -> str:
    backend = config.get("backend", "ollama")
    if backend == "ollama":
        import ollama
        resp = ollama.chat(
            model=config.get("model", "llama3.2"),
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp["message"]["content"]
    elif backend == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=config["api_key"])
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text
    raise ValueError(f"Unknown backend: {backend}")


def _detect_language(text: str) -> str:
    text_lower = text.lower()
    lang_keywords = {
        "python":     ["python", ".py", "django", "flask", "pandas", "numpy"],
        "javascript": ["javascript", "js", "node", "react", "express"],
        "typescript": ["typescript", "ts", ".ts"],
        "java":       ["java", "spring", "maven"],
        "html":       ["html", "webpage", "web page"],
        "css":        ["css", "stylesheet"],
        "bash":       ["bash", "shell script"],
        "sql":        ["sql", "query", "database", "select"],
        "cpp":        ["c++", "cpp"],
        "rust":       ["rust", ".rs"],
    }
    for lang, keywords in lang_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return lang
    return "python"


def _infer_filename(text: str, language: str) -> str:
    skip = {"create", "make", "write", "a", "an", "the", "for", "to", "that",
            "with", "and", "file", "code", "script", "in", "using", "function", "python"}
    words    = re.sub(r"[^a-zA-Z0-9 ]", "", text).split()
    keywords = [w.lower() for w in words if w.lower() not in skip][:3]
    base     = "_".join(keywords) if keywords else "generated"
    ext_map  = {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "java": ".java", "c": ".c", "cpp": ".cpp", "rust": ".rs",
        "go": ".go", "bash": ".sh", "shell": ".sh", "html": ".html",
        "css": ".css", "sql": ".sql",
    }
    return f"{base}{ext_map.get(language.lower(), '.py')}"