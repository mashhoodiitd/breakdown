# SciFact Dense Retriever

A reusable Python class for dense retrieval over the **SciFact** corpus. It embeds documents with OpenAI's `text-embedding-3-small`, caches the vectors in a NumPy file, and ranks documents using cosine similarity.

Dense embeddings can match paraphrases and conceptual meaning despite limited word overlap. BM25 can perform better for exact symbols, rare terms, ticker symbols, tax form codes, and regulation names.

Embedding documentation: https://platform.openai.com/docs/guides/embeddings

## 1. Project structure

```text
project/
├── scifact_dense_retriever.py
├── data/
│   └── scifact/
│       └── corpus.parquet
└── indexes/
    └── dense/
        └── embeddings.npy
```

The Parquet file must contain `_id` (document identifier) and `text` (document content). Empty text is replaced with `[empty document]` before embedding so vector rows remain aligned with document IDs.

## 2. Install dependencies

```bash
pip install numpy pandas openai python-dotenv tqdm pyarrow
```

Set your API key in the environment or a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Keep API keys out of source control.

## 3. Basic usage

```python
from scifact_dense_retriever import SciFactDenseRetriever

retriever = SciFactDenseRetriever()
retriever.load_corpus()
retriever.build_or_load_index()

results = retriever.search(
    "What is the relationship between alcohol and cancer risk?",
    k=5,
)

for doc_id, score in results:
    print(doc_id, score)
```

To include document text:

```python
for result in retriever.search_with_text("alcohol and cancer risk", k=5):
    print(result["doc_id"], result["score"])
    print(result["text"])
```

Or use the convenience printer:

```python
retriever.run("alcohol and cancer risk", k=5)
```

## 4. Configuration

```python
retriever = SciFactDenseRetriever(
    data_dir="data/scifact",
    index_dir="indexes/dense",
    model="text-embedding-3-small",
    api_key=None,       # Uses OPENAI_API_KEY by default
    batch_size=256,
)
```

- `data_dir`: directory containing `corpus.parquet`.
- `index_dir`: directory for the cached `embeddings.npy`.
- `model`: OpenAI embedding model.
- `api_key`: optional API key; defaults to `OPENAI_API_KEY`.
- `batch_size`: number of documents embedded per API request.

## 5. Class methods

| Method | Purpose |
|---|---|
| `load_corpus()` | Loads the SciFact Parquet corpus and prepares IDs and text. |
| `embed_batch(texts)` | Embeds one batch and returns a NumPy matrix. |
| `build_index(doc_texts=None, batch_size=None)` | Embeds documents in batches. |
| `save_index(embeddings=None)` | Saves vectors to `embeddings.npy`. |
| `load_index()` | Loads the cached vector matrix. |
| `build_or_load_index(force_rebuild=False)` | Loads the cache or creates it, then normalizes document vectors. |
| `search(query, k=10)` | Returns `(doc_id, cosine_similarity)` pairs. |
| `search_with_text(query, k=10)` | Returns IDs, scores, and document text. |
| `run(query, k=5)` | Prints and returns formatted search results. |

## 6. Cache and corpus updates

The first build embeds the corpus and saves the matrix to `indexes/dense/embeddings.npy`. Later runs reuse the cache.

If document contents change, rebuild the cache:

```python
retriever.build_or_load_index(force_rebuild=True)
```

The implementation checks that the cached vector count matches the current corpus size. If document text changes while the number of documents remains the same, this count check cannot detect the change; use `force_rebuild=True` or remove the cache.

## 7. Retrieval workflow

1. Load document IDs and text from `data/scifact/corpus.parquet`.
2. Replace empty document strings with a placeholder.
3. Embed documents in batches.
4. Save the embedding matrix as a NumPy `.npy` file.
5. Normalize document vectors once.
6. Embed and normalize each query.
7. Calculate cosine similarity using a dot product and return the highest-scoring documents.

The score is cosine similarity, not a probability or calibrated confidence score.

## 8. Dense retrieval versus BM25

Dense retrieval is useful when the query and document express the same concept with different wording. BM25 is often stronger when exact lexical matches matter, such as identifiers, rare terminology, and codes. A hybrid retrieval pipeline can combine both methods.
