"""Reusable BM25 sparse retriever using bm25s."""

from pathlib import Path
from typing import Optional

import bm25s
import pandas as pd


class BM25Retriever:
    """Load a Parquet corpus, build/save a BM25 index, and search it."""

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        index_dir: Optional[str | Path] = None,
        stopwords: str = "en",
    ) -> None:
        root = Path(__file__).parent
        self.data_dir = Path(data_dir) if data_dir is not None else root / "data" / "scifact"
        self.index_dir = Path(index_dir) if index_dir is not None else root / "indexes" / "bm25"
        self.stopwords = stopwords
        self.corpus: Optional[pd.DataFrame] = None
        self.doc_ids: list[str] = []
        self.doc_titles: list[str] = []
        self.doc_texts: list[str] = []
        self.retriever = None

    def load_corpus(self) -> pd.DataFrame:
        """Load corpus.parquet and prepare document IDs, titles, and text."""
        path = self.data_dir / "corpus.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Corpus Parquet file not found: {path}")

        self.corpus = pd.read_parquet(path)
        required = {"_id", "title", "text"}
        missing = required.difference(self.corpus.columns)
        if missing:
            raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

        self.doc_ids = self.corpus["_id"].astype(str).tolist()
        self.doc_titles = self.corpus["title"].fillna("").astype(str).tolist()
        self.doc_texts = self.corpus["text"].fillna("").astype(str).tolist()
        return self.corpus

    def build_index(self):
        """Tokenize document bodies and build the in-memory BM25 index."""
        if self.corpus is None:
            self.load_corpus()
        if not self.doc_texts:
            raise ValueError("The corpus is empty; cannot build a BM25 index.")

        print(f"Indexing {len(self.doc_texts)} documents with BM25...")
        tokens = bm25s.tokenize(self.doc_texts, stopwords=self.stopwords)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)
        return self.retriever

    def save_index(self) -> Path:
        """Save the index and its ordered document ID mapping."""
        if self.retriever is None:
            raise RuntimeError("BM25 index is not built. Call build_index() first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.index_dir))
        (self.index_dir / "doc_ids.txt").write_text(
            "\\n".join(self.doc_ids), encoding="utf-8"
        )
        return self.index_dir

    def load_index(self):
        """Load a saved BM25 index and matching document ID mapping."""
        ids_path = self.index_dir / "doc_ids.txt"
        if not ids_path.is_file():
            raise FileNotFoundError(f"Document ID mapping not found: {ids_path}")

        self.retriever = bm25s.BM25.load(str(self.index_dir))
        self.doc_ids = ids_path.read_text(encoding="utf-8").splitlines()
        return self.retriever

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (document ID, BM25 score) pairs for a query."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        if self.retriever is None:
            raise RuntimeError("BM25 index is not loaded or built.")

        query_tokens = bm25s.tokenize([query], stopwords=self.stopwords)
        indices, scores = self.retriever.retrieve(query_tokens, k=k)
        results = []
        for position, doc_index in enumerate(indices[0].tolist()):
            if 0 <= doc_index < len(self.doc_ids):
                results.append((self.doc_ids[doc_index], float(scores[0][position])))
        return results

    def search_with_text(self, query: str, k: int = 10) -> list[dict[str, object]]:
        """Return search results with document title and full text."""
        if self.corpus is None:
            self.load_corpus()
        assert self.corpus is not None

        results = []
        for doc_id, score in self.search(query, k):
            matches = self.corpus[self.corpus["_id"].astype(str) == doc_id]
            if not matches.empty:
                row = matches.iloc[0]
                results.append({
                    "doc_id": doc_id,
                    "score": score,
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                })
        return results

    def run(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Load corpus, build and save the index, then search and print results."""
        self.load_corpus()
        self.build_index()
        self.save_index()
        results = self.search(query, k=k)

        print(f"\nQuery: {query}\n")
        assert self.corpus is not None
        for rank, (doc_id, score) in enumerate(results, start=1):
            matches = self.corpus[self.corpus["_id"].astype(str) == doc_id]
            snippet = str(matches.iloc[0]["text"])[:160] if not matches.empty else ""
            print(f"{rank}. [{score:6.2f}] {doc_id}  {snippet}")
        return results


if __name__ == "__main__":
    example_query = "Do Antidepressants Cure or Create Abnormal Brain States?"
    BM25Retriever().run(example_query, k=5)
