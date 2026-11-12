"""Builds the real assistant (SentenceTransformers + Chroma + Gemini)."""
from .config import Config
from .pipeline import RAGAssistant


def build_assistant(config=None, need_llm=True, persist_dir=None):
    config = config or Config()
    from .embedder import SentenceTransformerEmbedder
    from .vector_store import ChromaStore
    llm = None
    if need_llm:
        from .llm import GeminiClient, get_api_key
        llm = GeminiClient(get_api_key(), config.gemini_model, config.temperature, config.max_retries)
    return RAGAssistant(ChromaStore(path=persist_dir), SentenceTransformerEmbedder(config.embedding_model),
                        llm, config)
