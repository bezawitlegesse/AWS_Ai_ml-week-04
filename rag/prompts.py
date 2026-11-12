"""Prompts live in the prompts/ folder as plain text files, never inside the code."""
import re
from pathlib import Path

from .config import ROOT

PROMPTS_DIR = ROOT / "prompts"
_PLACEHOLDER = re.compile(r"<<([A-Z_]+)>>")


def load_prompt(name, prompts_dir=PROMPTS_DIR):
    """Read prompts/<name>.txt."""
    path = Path(prompts_dir) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render(template, **values):
    """Fill <<PLACEHOLDER>> markers in ONE pass, so text inserted from a PDF is never re-scanned."""
    def substitute(match):
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Missing value for placeholder <<{key}>>")
        return str(values[key])
    return _PLACEHOLDER.sub(substitute, template)
