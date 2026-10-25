"""
Reusable downloader for the SciFact benchmark from the Hugging Face BEIR repositories.

The dataset consists of:
- corpus: documents to search over
- queries: user queries
- qrels: ground-truth (query_id, doc_id, relevance) judgments

Sources:
- https://huggingface.co/datasets/BeIR/scifact
- https://huggingface.co/datasets/BeIR/scifact-qrels/tree/main
- https://sites.google.com/view/scifact

Dependencies:
    pip install datasets pyarrow
"""

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset


class SciFactDownloader:
    """Download SciFact datasets and cache them as Parquet files."""

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        """
        Initialize the downloader.

        Args:
            data_dir: Directory in which Parquet files will be saved.
                Defaults to a ``scifact`` directory next to this Python file.
        """
        self.data_dir = (
            Path(data_dir) if data_dir is not None
            else Path(__file__).parent / "scifact"
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.corpus: Optional[Dataset] = None
        self.queries: Optional[Dataset] = None
        self.qrels: Optional[Dataset] = None

    def download(self) -> tuple[Dataset, Dataset, Dataset]:
        """
        Download/load the corpus, queries, and test qrels from Hugging Face.

        Returns:
            A tuple containing (corpus, queries, qrels).
        """
        self.corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
        self.queries = load_dataset("BeIR/scifact", "queries", split="queries")
        self.qrels = load_dataset("BeIR/scifact-qrels", split="test")
        return self.corpus, self.queries, self.qrels

    def export_parquet(self) -> dict[str, Path]:
        """
        Export the loaded datasets to Parquet files.

        Raises:
            RuntimeError: If download() has not been called successfully.
        Returns:
            Mapping of dataset names to the resulting file paths.
        """
        if self.corpus is None or self.queries is None or self.qrels is None:
            raise RuntimeError("Datasets are not loaded. Call download() first.")

        paths = {
            "corpus": self.data_dir / "corpus.parquet",
            "queries": self.data_dir / "queries.parquet",
            "qrels": self.data_dir / "qrels.parquet",
        }

        self.corpus.to_parquet(str(paths["corpus"]))
        self.queries.to_parquet(str(paths["queries"]))
        self.qrels.to_parquet(str(paths["qrels"]))
        return paths

    def run(self) -> dict[str, Path]:
        """Download all datasets and export them to Parquet."""
        self.download()
        paths = self.export_parquet()
        self.print_summary(paths)
        return paths

    def print_summary(self, paths: Optional[dict[str, Path]] = None) -> None:
        """Print row counts and output paths for the loaded datasets."""
        if self.corpus is None or self.queries is None or self.qrels is None:
            raise RuntimeError("Datasets are not loaded. Call download() first.")

        paths = paths or {
            "corpus": self.data_dir / "corpus.parquet",
            "queries": self.data_dir / "queries.parquet",
            "qrels": self.data_dir / "qrels.parquet",
        }

        print(f"Corpus:  {len(self.corpus):>6} docs    -> {paths['corpus']}")
        print(f"Queries: {len(self.queries):>6} queries -> {paths['queries']}")
        print(f"Qrels:   {len(self.qrels):>6} judgments -> {paths['qrels']}")


if __name__ == "__main__":
    SciFactDownloader().run()
