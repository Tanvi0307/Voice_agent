# 🎙️ Voice-Controlled Local AI Agent

A complete voice-to-action AI pipeline: speak a command → transcribe → classify intent → execute local tools → display results in a clean UI.

---

## 🏗️ Architecture :

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Audio Input │───▶│     STT      │───▶│ Intent Classify │───▶│ Tool Execute │
│  mic / file  │    │   Whisper    │    │  LLM (Ollama /  │    │ file / code  │
│             │    │  Groq / OAI  │    │   Claude API)   │    │  / summarize │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                                      │
                                                               ┌──────▼──────┐
                                                               │  Streamlit  │
                                                               │     UI      │
                                                               └─────────────┘
```

### Components

| Layer | File | Description |
|-------|------|-------------|
| STT | `stt/transcriber.py` | Whisper (local), Groq API, or OpenAI Whisper |
| Intent | `intent/classifier.py` | LLM-based intent + parameter extraction |
| Tools | `tools/file_ops.py` | Create files and folders |
| Tools | `tools/code_gen.py` | LLM code generation + file save |
| Tools | `tools/summarizer.py` | Text summarization |
| Agent | `agent/executor.py` | Orchestrates everything + session memory |
| UI | `app.py` | Streamlit frontend |

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/voice-ai-agent.git
cd voice-ai-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up the LLM backend

**Option A: Ollama (recommended, fully local)**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2        # or mistral, phi3, etc.
ollama serve                # starts local server on :11434
```

**Option B: Claude API**
- Get an API key from https://console.anthropic.com
- Enter it in the sidebar of the UI

### 4. Run the app
```bash
streamlit run app.py
```

---

## ⚙️ Configuration

All settings are in the **sidebar** of the UI:

| Setting | Options | Default |
|---------|---------|---------|
| STT Backend | faster-whisper, Groq API, OpenAI | faster-whisper |
| Whisper Model | tiny / base / small / medium | base |
| LLM Backend | Ollama, Claude API | Ollama |
| Ollama Model | any installed model | llama3.2 |
| Compound Commands | on/off | on |
| Human-in-the-Loop | on/off | on |
| Session Memory | on/off | on |

---

## 🎯 Supported Intents

| Intent | Example Commands |
|--------|-----------------|
| `write_code` | "Write a Python retry function", "Create a JavaScript fetch wrapper" |
| `create_file` | "Create a file called notes.txt", "Make a folder called src" |
| `summarize` | "Summarize this: [text...]", "Give me a summary of machine learning" |
| `general_chat` | "What is recursion?", "Explain Docker to me" |

---

## ✨ Bonus Features Implemented

- **Compound Commands**: Multiple intents in one command ("Write code and save it as main.py")
- **Human-in-the-Loop**: Confirmation prompt before any file operations
- **Graceful Degradation**: Error handling for bad audio, unknown intents
- **Session Memory**: Chat history preserved across commands in same session

---

## 🔒 Safety

All file creation is sandboxed to the `output/` folder. Path traversal attempts are sanitized.

---

## 🖥️ Hardware Notes

| Whisper Model | RAM Required | Speed |
|---------------|-------------|-------|
| tiny | ~1 GB | ~32x realtime |
| base | ~1 GB | ~16x realtime |
| small | ~2 GB | ~6x realtime |
| medium | ~5 GB | ~2x realtime |

If your machine is slow, use the **Groq API** STT backend — it's free-tier and extremely fast (~10x realtime). Set `STT Backend = Groq API` and paste your key from https://console.groq.com.
