"""
Dense retrieval for the SciFact corpus.

Embeds documents with OpenAI's text-embedding-3-small model, caches vectors
as a NumPy matrix, and retrieves documents using cosine similarity.
"""

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


class SciFactDenseRetriever:
    """Build and query a cached dense-vector index for SciFact."""

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        index_dir: Optional[str | Path] = None,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 256,
    ) -> None:
        load_dotenv()
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else self.base_dir / "data" / "scifact"
        self.index_dir = Path(index_dir) if index_dir else self.base_dir / "indexes" / "dense"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self.batch_size = batch_size
        self.model = model

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key to the constructor.")
        self.client = OpenAI(api_key=key)

        self.corpus = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_embeddings_normed: Optional[np.ndarray] = None
        self.corpus_path = self.data_dir / "corpus.parquet"
        self.embeddings_path = self.index_dir / "embeddings.npy"

    def load_corpus(self) -> pd.DataFrame:
        """Load corpus.parquet and prepare document IDs and text."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"SciFact corpus not found: {self.corpus_path}")
        corpus = pd.read_parquet(self.corpus_path)
        missing = {"_id", "text"} - set(corpus.columns)
        if missing:
            raise ValueError(f"Corpus is missing columns: {sorted(missing)}")

        self.corpus = corpus
        self.doc_ids = corpus["_id"].astype(str).tolist()
        # The embeddings API rejects empty strings; preserve row alignment.
        self.doc_texts = [
            str(value).strip() if pd.notna(value) and str(value).strip()
            else "[empty document]"
            for value in corpus["text"].tolist()
        ]
        return corpus

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of text and return a float32 matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)

    def build_index(
        self,
        doc_texts: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed the corpus in batches, displaying progress."""
        texts = list(doc_texts) if doc_texts is not None else self.doc_texts
        if not texts:
            raise ValueError("No document texts available. Call load_corpus() first.")
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        batches = []
        for start in tqdm(range(0, len(texts), size), desc="Embedding SciFact"):
            batches.append(self.embed_batch(texts[start:start + size]))
        self.doc_embeddings = np.vstack(batches).astype(np.float32, copy=False)
        return self.doc_embeddings

    def save_index(self, embeddings: Optional[np.ndarray] = None) -> Path:
        """Save the embedding matrix to indexes/dense/embeddings.npy."""
        matrix = embeddings if embeddings is not None else self.doc_embeddings
        if matrix is None:
            raise ValueError("No embeddings to save.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, np.asarray(matrix, dtype=np.float32))
        return self.embeddings_path

    def load_index(self) -> np.ndarray:
        """Load the cached NumPy embedding matrix."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embedding cache not found: {self.embeddings_path}")
        self.doc_embeddings = np.load(self.embeddings_path)
        return self.doc_embeddings

    def _normalize_embeddings(self) -> None:
        if self.doc_embeddings is None:
            raise ValueError("Embedding matrix has not been loaded.")
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_embeddings_normed = self.doc_embeddings / norms

    def build_or_load_index(self, force_rebuild: bool = False) -> np.ndarray:
        """Load cached vectors or build and cache them if needed."""
        if self.corpus is None:
            self.load_corpus()

        if self.embeddings_path.exists() and not force_rebuild:
            print(f"Loading cached embeddings from {self.embeddings_path}")
            embeddings = self.load_index()
            if len(embeddings) != len(self.doc_ids):
                raise ValueError(
                    "Cached vector count differs from the current corpus. "
                    "Rebuild with force_rebuild=True."
                )
        else:
            print(f"Embedding {len(self.doc_texts)} SciFact documents with {self.model}...")
            embeddings = self.build_index()
            self.save_index(embeddings)

        self._normalize_embeddings()
        return embeddings

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (document_id, cosine_similarity) pairs."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")
        if self.doc_embeddings_normed is None:
            self.build_or_load_index()

        query_vector = self.embed_batch([query])[0]
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query embedding has zero magnitude.")
        query_vector = query_vector / norm

        scores = self.doc_embeddings_normed @ query_vector
        indices = np.argsort(-scores)[:min(k, len(scores))]
        return [(self.doc_ids[i], float(scores[i])) for i in indices]

    def search_with_text(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k results with document text and similarity score."""
        if self.corpus is None:
            self.load_corpus()
        text_by_id = dict(zip(self.doc_ids, self.doc_texts))
        return [
            {"doc_id": doc_id, "score": score, "text": text_by_id.get(doc_id, "")}
            for doc_id, score in self.search(query, k)
        ]

    def run(self, query: str, k: int = 5) -> list[dict]:
        """Print and return the top-k search results."""
        results = self.search_with_text(query, k)
        print(f"\\nQuery: {query}\\n")
        for rank, result in enumerate(results, 1):
            print(
                f"{rank}. [{result['score']:.3f}] "
                f"{result['doc_id']} {result['text']}\\n"
            )
        return results


if __name__ == "__main__":
    retriever = SciFactDenseRetriever()
    retriever.load_corpus()
    retriever.build_or_load_index()
    retriever.run("Where should I park my rainy-day fund?", k=5)
