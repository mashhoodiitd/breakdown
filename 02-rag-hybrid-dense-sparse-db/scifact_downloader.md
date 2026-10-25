# SciFact Downloader

A reusable Python class for downloading the SciFact benchmark datasets from the Hugging Face BEIR repositories and exporting them as Parquet files.

## Dataset components

- **Corpus** — documents to search over.
- **Queries** — user queries.
- **Qrels** — ground-truth query/document relevance judgments.

Sources:
- [SciFact dataset on Hugging Face](https://huggingface.co/datasets/BeIR/scifact)
- [SciFact qrels on Hugging Face](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/main)
- [SciFact project site](https://sites.google.com/view/scifact)

## Requirements

Install the dependencies:

```bash
pip install datasets pyarrow
```

## Usage

### Run the script

```bash
python scifact_downloader.py
```

This downloads/loads the three dataset components and exports them to a `scifact` directory next to the script.

### Use the class from another Python file

```python
from scifact_downloader import SciFactDownloader

downloader = SciFactDownloader()
downloader.run()
```

### Choose a custom output directory

```python
from scifact_downloader import SciFactDownloader

downloader = SciFactDownloader(data_dir="./data/scifact")
downloader.run()
```

## Class reference

### `SciFactDownloader(data_dir=None)`

Initializes the downloader and creates the output directory if it does not already exist.

- `data_dir` *(optional)*: output directory. Defaults to a `scifact` directory beside `scifact_downloader.py`.

### `download()`

Loads the datasets from Hugging Face and returns a tuple in this order:

1. `corpus`
2. `queries`
3. `qrels`

The loaded datasets are also stored on the instance as `downloader.corpus`, `downloader.queries`, and `downloader.qrels`.

### `export_parquet()`

Writes the loaded datasets to Parquet files and returns a dictionary mapping dataset names to their file paths. Call `download()` first; otherwise, the method raises `RuntimeError`.

### `run()`

Runs the complete workflow: download/load datasets, export them to Parquet, print a summary, and return the output paths.

### `print_summary(paths=None)`

Prints the row counts and output paths for the loaded datasets. Raises `RuntimeError` if the datasets have not been loaded.

## Output files

The default output directory contains:

```text
scifact/
├── corpus.parquet
├── queries.parquet
└── qrels.parquet
```

## Example output

The script prints a summary in this general format:

```text
Corpus:   <count> docs    -> <path>/corpus.parquet
Queries:  <count> queries -> <path>/queries.parquet
Qrels:    <count> judgments -> <path>/qrels.parquet
```

The exact counts and paths depend on the downloaded dataset and the configured output directory.

## Notes

- The script requires an internet connection the first time it downloads the datasets.
- Parquet export uses the `pyarrow` dependency.
- This documentation describes the current `SciFactDownloader` class interface.
