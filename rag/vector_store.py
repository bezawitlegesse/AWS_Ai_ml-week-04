"""Vector stores. ChromaStore is used by the app; InMemoryStore (NumPy only) is used by the offline tests."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    id: str
    text: str
    page: int
    distance: float        # cosine distance: 0 = same meaning, bigger = less similar


class InMemoryStore:
    def __init__(self):
        self.reset()

    def reset(self):
        self._ids, self._docs, self._pages, self._matrix = [], [], [], None

    def add(self, ids, documents, metadatas, embeddings):
        self._ids += list(ids)
        self._docs += list(documents)
        self._pages += [int(m["page"]) for m in metadatas]
        new = np.asarray(embeddings, dtype=float)
        new = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def count(self):
        return len(self._ids)

    def query(self, embedding, k):
        if self._matrix is None:
            return []
        vector = np.asarray(embedding, dtype=float)
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        distances = 1.0 - self._matrix @ vector
        order = np.argsort(distances)[:k]
        return [Hit(self._ids[i], self._docs[i], self._pages[i], float(distances[i])) for i in order]


class ChromaStore:
    def __init__(self, name="pdf_chunks", path=None):
        import chromadb
        self._client = chromadb.PersistentClient(path=path) if path else chromadb.Client()
        self._name = name
        self.reset()

    def reset(self):
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._collection = self._client.create_collection(name=self._name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, documents, metadatas, embeddings, batch_size=500):
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(ids=list(ids[start:end]), documents=list(documents[start:end]),
                                 metadatas=list(metadatas[start:end]), embeddings=list(embeddings[start:end]))

    def count(self):
        return self._collection.count()

    def query(self, embedding, k):
        n = min(k, self.count())
        if n == 0:
            return []
        result = self._collection.query(query_embeddings=[embedding], n_results=n,
                                        include=["documents", "metadatas", "distances"])
        return [Hit(i, d, int(m["page"]), float(dist)) for i, d, m, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])]
