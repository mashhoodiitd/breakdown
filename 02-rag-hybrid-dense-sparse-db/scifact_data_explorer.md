# SciFact Data Explorer

A reusable Python class for loading and exploring the SciFact benchmark's corpus, queries, and relevance judgments from Parquet files.

## Dataset components

- **Corpus** — scientific documents to search over.
- **Queries** — science questions.
- **Qrels** — relevance judgments connecting query IDs to relevant corpus document IDs.

The script inspects sample rows, filters queries to those present in the qrels, and displays one query with its relevant document IDs.

More information: [BEIR GitHub repository](https://github.com/beir-cellar/beir)

## Requirements

Install the dependencies:

```bash
pip install pandas pyarrow
```

## Expected input files

By default, the class looks for these files relative to `scifact_data_explorer.py`:

```text
data/
└── scifact/
    ├── corpus.parquet
    ├── queries.parquet
    └── qrels.parquet
```

These files can be created by the SciFact downloader script. To use a different location, pass a custom `data_dir`.

## Usage

### Run the complete workflow

```bash
python scifact_data_explorer.py
```

### Import and use the class

```python
from scifact_data_explorer import SciFactDataExplorer

explorer = SciFactDataExplorer()
results = explorer.run()
```

### Specify a custom data directory

```python
from scifact_data_explorer import SciFactDataExplorer

explorer = SciFactDataExplorer(data_dir="./my_data/scifact")
results = explorer.run()
```

## Class reference

### `SciFactDataExplorer(data_dir=None)`

Initializes the explorer.

- `data_dir` *(optional)*: directory containing the three Parquet files.
- Default: `data/scifact` beside the Python class file.

### `load_data()`

Loads the three Parquet files into pandas DataFrames and returns a tuple `(corpus, queries, qrels)`. Raises `FileNotFoundError` if any required file is missing.

### `inspect_first_rows()`

Prints the first corpus document's `text`, the first query's `text`, the first qrels row, and the number of qrels rows. Call `load_data()` first.

### `filter_test_queries()`

Filters `queries` to rows whose `_id` occurs in the qrels `query-id` column. IDs are converted to strings for matching. Prints and returns the filtered DataFrame, also storing it as `explorer.test_queries`.

### `inspect_example_query(position=0)`

Prints one filtered query and the list of `corpus-id` values judged relevant to that query.

- `position`: zero-based row position in the filtered query DataFrame.
- Returns a dictionary containing the selected query row and matching qrels rows.

The default selects the first filtered query; it does not randomly select a query.

### `run()`

Runs the full workflow:

1. Loads the three Parquet files.
2. Prints example rows.
3. Filters queries to those with qrels.
4. Prints the number of matching queries.
5. Inspects the first filtered query and its relevant document IDs.

Returns a dictionary containing the loaded DataFrames, filtered queries, and example results.

## Notes

- This script explores existing Parquet files; it does not download the dataset or write new files.
- If the Parquet files are missing, run the downloader first or provide the correct `data_dir`.
