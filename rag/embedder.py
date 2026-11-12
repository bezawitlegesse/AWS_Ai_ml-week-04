"""Turns text into vectors (lists of numbers) that capture meaning."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """Return one normalised vector (list of floats) per text."""
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()
