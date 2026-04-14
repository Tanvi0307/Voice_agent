"""
app.py — Streamlit UI for Voice-Controlled Local AI Agent
Run: streamlit run app.py
"""

import tempfile
import datetime
import json
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import streamlit as st

from agent.executor import AgentExecutor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
.pipeline-card {
    background: #13131a; border: 1px solid #2a2a3a;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    margin: 0.5rem 0; position: relative; overflow: hidden;
}
.pipeline-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; border-radius: 3px 0 0 3px;
}
.card-stt::before    { background: #6366f1; }
.card-intent::before { background: #f59e0b; }
.card-action::before { background: #10b981; }
.card-output::before { background: #3b82f6; }
.card-label {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 0.4rem; opacity: 0.5;
}
.card-value { font-size: 1rem; color: #e8e8f0; line-height: 1.6; }
.intent-badge {
    display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
    background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; margin-right: 4px;
}
.header-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 50%, #3b82f6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-sub {
    color: #555570; font-size: 0.9rem; letter-spacing: 0.08em;
    text-transform: uppercase; font-weight: 600;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
}
.file-saved-box {
    background: #0d1f0d; border: 1px solid #1a3a1a;
    border-radius: 10px; padding: 0.8rem 1.2rem; margin: 0.5rem 0;
    display: flex; align-items: center; gap: 10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR  = Path(__file__).parent / "output"
HISTORY_FILE = Path(__file__).parent / "history.json"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT HISTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_history() -> list:
    """Load history from disk."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(history: list):
    """Save history to disk."""
    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception:
        pass


def append_history(item: dict):
    """Append one item to persistent history."""
    history = load_history()
    history.append(item)
    save_history(history)


# ══════════════════════════════════════════════════════════════════════════════
# FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_language(filename: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".html": "html", ".css": "css", ".json": "json", ".sql": "sql",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
        ".go": "go", ".rs": "rust", ".rb": "ruby", ".sh": "bash",
        ".md": "markdown", ".yaml": "yaml", ".xml": "xml",
    }
    return ext_map.get(Path(filename).suffix.lower(), "text")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "recorded_audio": None,
        "agent": AgentExecutor(),
        "awaiting_confirm": False,
        "pending_transcript": None,
        "pending_intent_result": None,
        "pending_llm_config": None,
        "pending_memory": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
agent: AgentExecutor = st.session_state.agent


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _run_pipeline(audio_path: str, stt_config: dict, llm_config: dict, hitl: bool, memory: bool):
    with st.spinner("🔊 Transcribing audio…"):
        try:
            transcript = agent.transcribe(audio_path, stt_config)
        except Exception as e:
            st.error(f"❌ Transcription error: {e}")
            return
    _show_results(transcript, llm_config, hitl, memory)


def _run_pipeline_from_text(text: str, llm_config: dict, hitl: bool, memory: bool):
    _show_results(text, llm_config, hitl, memory, skip_stt=True)


def _show_results(transcript: str, llm_config: dict, hitl: bool, memory: bool, skip_stt: bool = False):
    st.markdown("#### 🔍 Pipeline Results")

    # Transcript card
    label = "⌨️ Input Text" if skip_stt else "🎙️ Transcription"
    st.markdown(f"""
    <div class="pipeline-card card-stt">
        <div class="card-label">{label}</div>
        <div class="card-value">{transcript}</div>
    </div>""", unsafe_allow_html=True)

    # Intent detection
    with st.spinner("🧠 Detecting intent…"):
        try:
            intent_result = agent.detect_intent(transcript, llm_config)
        except Exception as e:
            st.error(f"❌ Intent detection failed: {e}")
            st.info("💡 Make sure Ollama is running (ollama serve) or your API key is correct.")
            return

    badges = " ".join(f'<span class="intent-badge">{i}</span>' for i in intent_result["intents"])
    st.markdown(f"""
    <div class="pipeline-card card-intent">
        <div class="card-label">🎯 Detected Intent</div>
        <div class="card-value">{badges}</div>
    </div>""", unsafe_allow_html=True)

    # HITL
    needs_file_op = any(i in ["create_file", "write_code"] for i in intent_result["intents"])
    if hitl and needs_file_op:
        st.warning("⚠️ This will create/modify files in output/. Confirm?")
        col_y, col_n, _ = st.columns([1, 1, 5])
        confirmed = col_y.button("✅ Confirm", key="hitl_confirm")
        cancelled = col_n.button("❌ Cancel",  key="hitl_cancel")
        if cancelled:
            st.info("Action cancelled.")
            return
        if not confirmed:
            st.info("👆 Click Confirm to proceed.")
            return

    # Execute
    with st.spinner("⚙️ Executing…"):
        try:
            execution = agent.execute(transcript, intent_result, llm_config, use_memory=memory)
        except Exception as e:
            st.error(f"❌ Execution error: {e}")
            import traceback
            with st.expander("🐛 Full traceback"):
                st.code(traceback.format_exc())
            return

    # Action card
    st.markdown(f"""
    <div class="pipeline-card card-action">
        <div class="card-label">⚙️ Action Taken</div>
        <div class="card-value">{execution.get('action_description', 'No action')}</div>
    </div>""", unsafe_allow_html=True)

    # Output card
    st.markdown(f"""
    <div class="pipeline-card card-output">
        <div class="card-label">✅ Output</div>
        <div class="card-value">{execution.get('summary', '')}</div>
    </div>""", unsafe_allow_html=True)

    # ── File saved confirmation + auto-save notice ─────────────────────────────
    if execution.get("file_path"):
        file_path = Path(execution["file_path"])
        rel_path  = execution.get("relative_path", file_path.name)

        # Green auto-saved box
        st.markdown(f"""
        <div class="file-saved-box">
            <span style="font-size:1.4rem">✅</span>
            <div>
                <div style="color:#4ade80;font-weight:700;font-size:0.95rem">
                    File auto-saved to output folder
                </div>
                <div style="color:#86efac;font-size:0.8rem;font-family:monospace">
                    📁 {rel_path}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Preview + download
        content = execution.get("content", "")
        if content:
            with st.expander("👁️ View Generated File", expanded=True):
                st.code(content, language=execution.get("language", "python"))
                # Download button inside preview
                st.download_button(
                    label=f"⬇️ Download {file_path.name}",
                    data=content.encode("utf-8"),
                    file_name=file_path.name,
                    mime="application/octet-stream",
                    key=f"dl_{file_path.name}_{datetime.datetime.now().timestamp()}",
                )

    # Chat response
    if execution.get("chat_response"):
        st.markdown("---")
        st.info(f"💬 {execution['chat_response']}")

    # ── Save to persistent history ─────────────────────────────────────────────
    history_item = {
        "transcript": transcript,
        "intent":     ", ".join(intent_result["intents"]),
        "intents":    intent_result["intents"],
        "action":     execution.get("action_description", ""),
        "output":     execution.get("content", ""),
        "file":       execution.get("relative_path", ""),
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_history(history_item)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="header-title">🎙️ Voice AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Speak → Transcribe → Understand → Execute</div>', unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (hardcoded defaults — change here or expose in sidebar)
# ══════════════════════════════════════════════════════════════════════════════
stt_config = {"backend": "local", "model": "tiny"}
llm_config = {"backend": "ollama", "model": "llama3.2"}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Features")
    enable_hitl   = st.checkbox("Human-in-the-Loop", value=False)
    enable_memory = st.checkbox("Session Memory",    value=True)

    st.markdown("---")
    st.markdown("### 📂 Output Files")
    files = sorted(
        [f for f in OUTPUT_DIR.glob("*") if f.is_file()],
        key=lambda x: x.stat().st_mtime, reverse=True
    )
    if files:
        for f in files:
            with st.expander(f"📄 {f.name}"):
                try:
                    content = f.read_text(encoding="utf-8")
                    st.code(content[:1000], language=get_language(f.name))
                except Exception:
                    st.warning("Binary file")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇️ Download", data=f.read_bytes(),
                        file_name=f.name, mime="application/octet-stream",
                        key=f"sb_dl_{f.name}"
                    )
                with col2:
                    if st.button("🗑️ Delete", key=f"sb_del_{f.name}"):
                        f.unlink()
                        st.rerun()
    else:
        st.caption("No files yet")

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear Files", disabled=not files):
        for f in OUTPUT_DIR.glob("*"):
            if f.is_file():
                f.unlink()
        st.rerun()
    if col2.button("🗑️ Clear History"):
        save_history([])
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS  — Agent  |  History
# ══════════════════════════════════════════════════════════════════════════════
tab_agent, tab_history = st.tabs(["🤖 Agent", "📜 History"])


# ── AGENT TAB ──────────────────────────────────────────────────────────────────
with tab_agent:
    input_tab_mic, input_tab_upload, input_tab_text, input_tab_summarize = st.tabs([
        "🎤 Microphone", "📁 Upload Audio", "⌨️ Type Text", "📝 Summarize"
    ])

    with input_tab_mic:
        st.markdown("#### Record from Microphone")
        col1, col2, col3 = st.columns([1, 1, 2])
        duration = col1.slider("Seconds", 3, 60, 10)
        if col1.button("⏺️ Record"):
            st.info(f"🔴 Recording for {duration}s…")
            audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, 16000)
                st.session_state.recorded_audio = f.name
            st.success("✅ Done!")
            st.rerun()
        if st.session_state.recorded_audio:
            st.audio(st.session_state.recorded_audio)
            if col2.button("▶️ Process"):
                _run_pipeline(st.session_state.recorded_audio, stt_config, llm_config, enable_hitl, enable_memory)
            if col3.button("🗑️ Clear Recording"):
                st.session_state.recorded_audio = None
                st.rerun()

    with input_tab_upload:
        st.markdown("#### Upload Audio File")
        uploaded = st.file_uploader("Choose .wav / .mp3 / .ogg / .m4a", type=["wav", "mp3", "ogg", "m4a"])
        if uploaded:
            st.audio(uploaded)
            if st.button("▶️ Process Audio"):
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
                    f.write(uploaded.read())
                    _run_pipeline(f.name, stt_config, llm_config, enable_hitl, enable_memory)

    with input_tab_text:
        st.markdown("#### Type Your Command")
        text_input = st.text_area(
            "Command",
            placeholder='"Create a python file to add two numbers"',
            height=100,
        )
        if st.button("▶️ Execute Command"):
            if text_input.strip():
                _run_pipeline_from_text(text_input.strip(), llm_config, enable_hitl, enable_memory)
            else:
                st.warning("Please enter a command.")

    with input_tab_summarize:
        st.markdown("#### Summarize Text")
        text_to_summarize = st.text_area(
            "Paste text to summarize:",
            placeholder="Enter the text you want summarized…",
            height=150,
        )
        col1, col2 = st.columns(2)
        save_option = col1.checkbox("Save as file?", value=False)
        filename    = col2.text_input("Filename:", placeholder="summary.txt", disabled=not save_option)

        if st.button("📝 Summarize"):
            if text_to_summarize.strip():
                intent_result = {
                    "intents": ["summarize"],
                    "params": {
                        "text_to_summarize": text_to_summarize,
                        "filename": filename if save_option and filename else None,
                    },
                }
                with st.spinner("Summarizing…"):
                    try:
                        execution = agent.execute(
                            f"Summarize: {text_to_summarize[:100]}",
                            intent_result, llm_config, use_memory=enable_memory
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.stop()

                st.markdown("**📋 Summary:**")
                st.info(execution.get("content", ""))

                if execution.get("file_path"):
                    file_path = Path(execution["file_path"])
                    st.markdown(f"""
                    <div class="file-saved-box">
                        <span style="font-size:1.4rem">✅</span>
                        <div>
                            <div style="color:#4ade80;font-weight:700">Summary auto-saved</div>
                            <div style="color:#86efac;font-size:0.8rem;font-family:monospace">
                                📁 {execution.get('relative_path', file_path.name)}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Save to history
                append_history({
                    "transcript": f"Summarize text ({len(text_to_summarize)} chars)",
                    "intent":     "summarize",
                    "intents":    ["summarize"],
                    "action":     "Summarized text",
                    "output":     execution.get("content", ""),
                    "file":       execution.get("relative_path", ""),
                    "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            else:
                st.warning("Please enter text to summarize.")


# ── HISTORY TAB  (separate component, loads from disk) ─────────────────────────
with tab_history:
    st.markdown("### 📜 Full Session History")
    st.caption("All actions are saved to `history.json` and persist across restarts.")

    history = load_history()

    if not history:
        st.info("No history yet. Execute some commands to see them here.")
    else:
        # Stats row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Actions", len(history))
        col2.metric("Files Created", sum(1 for h in history if h.get("file")))
        col3.metric("Last Active", history[-1]["timestamp"].split(" ")[0] if history else "—")

        st.markdown("---")

        # Search / filter
        search = st.text_input("🔍 Search history", placeholder="Filter by keyword…")
        intent_filter = st.multiselect(
            "Filter by intent",
            ["write_code", "create_file", "summarize", "general_chat"],
            default=[],
        )

        filtered = list(reversed(history))  # newest first
        if search:
            filtered = [h for h in filtered if search.lower() in h.get("transcript", "").lower()
                        or search.lower() in h.get("output", "").lower()]
        if intent_filter:
            filtered = [h for h in filtered if any(i in h.get("intents", []) for i in intent_filter)]

        st.caption(f"Showing {len(filtered)} of {len(history)} records")
        st.markdown("---")

        for i, item in enumerate(filtered):
            intent_label = item.get("intent", "unknown")
            ts           = item.get("timestamp", "")
            cmd_preview  = item.get("transcript", "")[:60]

            with st.expander(f"🕐 {ts}  ·  {intent_label}  ·  {cmd_preview}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Command:** {item.get('transcript', '')}")
                    st.markdown(f"**Action:** {item.get('action', '')}")
                with col2:
                    if item.get("file"):
                        st.markdown(f"**File:** `{item['file']}`")
                        # Check if file still exists
                        full_path = Path(__file__).parent / item["file"]
                        if full_path.exists():
                            st.download_button(
                                "⬇️ Download",
                                data=full_path.read_bytes(),
                                file_name=full_path.name,
                                mime="application/octet-stream",
                                key=f"hist_dl_{i}_{item['timestamp']}",
                            )
                        else:
                            st.caption("⚠️ File deleted")

                if item.get("output"):
                    st.code(item["output"][:500], language=get_language(item.get("file", ".txt")))

        st.markdown("---")
        if st.button("🗑️ Clear All History", type="primary"):
            save_history([])
            st.rerun()