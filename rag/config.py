"""All tunable settings in one place. Environment variables override the model names."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.getenv(name) or default


@dataclass
class Config:
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.8-flash"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chunk_size: int = 500          # characters per chunk
    chunk_overlap: int = 50        # characters shared by neighbouring chunks
    top_k: int = 3                 # chunks retrieved per question
    max_distance: float = 0.80     # cosine distance; above this a chunk counts as "not relevant"
    max_question_chars: int = 500
    max_repairs: int = 2           # extra tries when the JSON answer is invalid
    max_retries: int = 4           # retries for temporary API errors (429, 503, ...)
    temperature: float = 0.2
    history_turns: int = 6         # past turns kept for follow-up questions
