"""
tools/file_ops.py
Handles file and folder creation.
ALL operations are sandboxed to the output/ directory.
"""

import os
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_file(filename: str, content: str = "", description: str = "") -> dict:
    """
    Create a file inside output/.

    Args:
        filename: Target filename (e.g. "notes.txt", "src/main.py")
        content:  File body (empty string creates a blank file)
        description: Human-readable description of what was done

    Returns:
        dict with keys: success, file_path, message
    """
    # Safety: strip leading slashes / dots to stay inside output/
    safe_name = _sanitize(filename)
    target = OUTPUT_DIR / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(content, encoding="utf-8")

    return {
        "success": True,
        "file_path": str(target),
        "relative_path": f"output/{safe_name}",
        "message": f"Created file: output/{safe_name}",
        "content": content,
    }


def create_folder(folder_name: str) -> dict:
    """Create a folder inside output/."""
    safe_name = _sanitize(folder_name)
    target = OUTPUT_DIR / safe_name
    target.mkdir(parents=True, exist_ok=True)
    return {
        "success": True,
        "file_path": str(target),
        "message": f"Created folder: output/{safe_name}",
    }


def list_output_files() -> list:
    """Return a list of all files currently in output/."""
    return [str(p.relative_to(OUTPUT_DIR)) for p in OUTPUT_DIR.rglob("*") if p.is_file()]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _sanitize(name: str) -> str:
    """Remove any path traversal attempts."""
    parts = Path(name).parts
    safe = Path(*[p for p in parts if p not in ("", ".", "..", "/", "\\")])
    return str(safe)
