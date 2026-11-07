# BM25 Retriever

A reusable Python class for sparse, keyword-based retrieval using [bm25s](https://github.com/xhluca/bm25s).

The class loads a corpus from Parquet, tokenizes document text, builds a BM25 index, saves the index and document ID mapping, and supports search.

> **Dataset note:** The supplied script describes FiQA in its opening docstring, but points to `data/scifact` and expects `_id`, `title`, and `text` columns. This refactor preserves those configured paths and columns. Use `data_dir` to point to another compatible corpus.

## How BM25 works

BM25 ranks documents using query-term frequency and the rarity of each term across the corpus.

- It is effective for exact terms and rare words.
- It is keyword-based and does not use embeddings or a language model.
- It may miss relevant documents when queries and documents use different wording or paraphrases.

## Requirements

```bash
pip install bm25s pandas pyarrow
```

## Expected input

By default, provide:

```text
data/
└── scifact/
    └── corpus.parquet
```

The Parquet file must contain `_id`, `title`, and `text` columns. The index is built over the `text` column.

## Usage

### Build, save, and search

```python
from fiqa_bm25_retriever import BM25Retriever

retriever = BM25Retriever()
results = retriever.run(
    "Do Antidepressants Cure or Create Abnormal Brain States?",
    k=5,
)
```

### Configure paths

```python
retriever = BM25Retriever(
    data_dir="./data/scifact",
    index_dir="./indexes/my_bm25",
)
retriever.load_corpus()
retriever.build_index()
retriever.save_index()

results = retriever.search("antidepressants and brain states", k=10)
```

### Load an existing index

```python
retriever = BM25Retriever()
retriever.load_index()
results = retriever.search("antidepressants and brain states", k=5)
```

To retrieve document titles and text along with scores, use `search_with_text()`; it loads the corpus if necessary.

## Class reference

### `BM25Retriever(data_dir=None, index_dir=None, stopwords="en")`

Initializes the retriever.

- `data_dir`: directory containing `corpus.parquet`; defaults to `data/scifact` beside this file.
- `index_dir`: directory for the saved BM25 index and `doc_ids.txt`; defaults to `indexes/bm25` beside this file.
- `stopwords`: stopword option passed to `bm25s.tokenize`; defaults to English stopwords.

### `load_corpus()`

Reads the Parquet corpus and prepares document ID, title, and text lists. Checks that the required columns are present.

### `build_index()`

Tokenizes document bodies and builds the in-memory `bm25s.BM25` index.

### `save_index()`

Saves the index using bm25s and writes document IDs, in index order, to `doc_ids.txt`. The mapping must remain aligned with the saved index.

### `load_index()`

Loads a previously saved index and its document ID mapping.

### `search(query, k=10)`

Returns up to `k` results as `(doc_id, score)` tuples, ordered by retrieval rank.

### `search_with_text(query, k=10)`

Returns result dictionaries with `doc_id`, `score`, `title`, and `text`.

### `run(query, k=5)`

Runs the complete workflow: load corpus, build index, save index, search, and print ranked text snippets.

## Generated index files

The index directory contains bm25s-managed index files and:

```text
indexes/
└── bm25/
    ├── ...          # files created by bm25s
    └── doc_ids.txt  # document IDs in index order
```

## Important considerations

- Keep `doc_ids.txt` paired with the exact index it was generated with. A mismatch can map retrieved positions to the wrong documents.
- Rebuild and save the index when the corpus changes.
- This implementation provides sparse BM25 retrieval; it does not implement semantic search, hybrid retrieval, or reranking.
