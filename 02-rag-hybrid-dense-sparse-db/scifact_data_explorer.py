"""
Explore the SciFact benchmark's corpus, queries, and relevance judgments.

SciFact is a scientific question-answering benchmark:
- corpus: scientific documents
- queries: questions
- qrels: ground-truth query/document relevance judgments

Expected input files by default:
    data/scifact/corpus.parquet
    data/scifact/queries.parquet
    data/scifact/qrels.parquet

More information: https://github.com/beir-cellar/beir
"""

from pathlib import Path
from typing import Optional

import pandas as pd


class SciFactDataExplorer:
    """Load and inspect SciFact datasets stored as Parquet files."""

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        """Initialize the explorer.

        Args:
            data_dir: Directory containing corpus.parquet, queries.parquet,
                and qrels.parquet. Defaults to ./data/scifact beside this file.
        """
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).parent / "data" / "scifact"
        )
        self.corpus: Optional[pd.DataFrame] = None
        self.queries: Optional[pd.DataFrame] = None
        self.qrels: Optional[pd.DataFrame] = None
        self.test_queries: Optional[pd.DataFrame] = None

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the corpus, queries, and qrels Parquet files."""
        corpus_path = self.data_dir / "corpus.parquet"
        queries_path = self.data_dir / "queries.parquet"
        qrels_path = self.data_dir / "qrels.parquet"

        for path in (corpus_path, queries_path, qrels_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required dataset file not found: {path}")

        self.corpus = pd.read_parquet(corpus_path)
        self.queries = pd.read_parquet(queries_path)
        self.qrels = pd.read_parquet(qrels_path)
        return self.corpus, self.queries, self.qrels

    def inspect_first_rows(self) -> None:
        """Print the first corpus text, query text, and qrels row."""
        self._ensure_data_loaded()
        assert self.corpus is not None
        assert self.queries is not None
        assert self.qrels is not None

        if self.corpus.empty or self.queries.empty or self.qrels.empty:
            raise ValueError("One or more loaded datasets are empty.")

        print(self.corpus["text"].iloc[0])
        print(self.queries["text"].iloc[0])
        print(self.qrels.iloc[0])
        print(f"All the queries with qrels: {len(self.qrels)}")

    def filter_test_queries(self) -> pd.DataFrame:
        """Keep queries whose IDs occur in the qrels query-id column."""
        self._ensure_data_loaded()
        assert self.queries is not None
        assert self.qrels is not None

        test_query_ids = set(self.qrels["query-id"].astype(str))
        self.test_queries = self.queries[
            self.queries["_id"].astype(str).isin(test_query_ids)
        ]
        print(f"Number of queries that have qrels: {len(self.test_queries)}")
        return self.test_queries

    def inspect_example_query(self, position: int = 0) -> dict:
        """Print a filtered query and the corpus IDs marked relevant to it.

        Args:
            position: Zero-based row position in the filtered query DataFrame.

        Returns:
            A dictionary with the selected query row and its matching qrels rows.
        """
        self._ensure_data_loaded()
        if self.test_queries is None:
            self.filter_test_queries()

        assert self.test_queries is not None
        assert self.qrels is not None

        if self.test_queries.empty:
            raise ValueError("No queries matched the query IDs in qrels.")
        if position < 0 or position >= len(self.test_queries):
            raise IndexError(
                f"position must be between 0 and {len(self.test_queries) - 1}."
            )

        query = self.test_queries.iloc[position]
        relevant = self.qrels[
            self.qrels["query-id"].astype(str) == str(query["_id"])
        ]
        print("\nExample query")
        print(f"  _id:  {query['_id']}")
        print(f"  text: {query['text']}")
        print(f"\n  Relevant docs for this question: {list(relevant['corpus-id'])}")
        return {"query": query, "relevant_qrels": relevant}

    def run(self) -> dict:
        """Run the complete data exploration workflow."""
        self.load_data()
        self.inspect_first_rows()
        test_queries = self.filter_test_queries()
        example = self.inspect_example_query()
        return {
            "corpus": self.corpus,
            "queries": self.queries,
            "qrels": self.qrels,
            "test_queries": test_queries,
            "example": example,
        }

    def _ensure_data_loaded(self) -> None:
        """Raise an informative error if load_data() has not been called."""
        if self.corpus is None or self.queries is None or self.qrels is None:
            raise RuntimeError("Data is not loaded. Call load_data() first.")


if __name__ == "__main__":
    SciFactDataExplorer().run()
