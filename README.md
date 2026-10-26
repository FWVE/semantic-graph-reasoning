# Semantic Graph Reasoning

A Coordinator–Inspector framework for reasoning over text-rich graphs, with
**SemGraph-Bench: 450 queries across seven graphs and nine structural families**.

More coming soon.

## Abstract

Semantic Graph Reasoning combines graph computation with interpretation of node
and edge text. A Coordinator plans graph operations and semantic investigations,
while independent Inspectors examine source evidence. Shared judgments and
dependency tracking support recovery of complete qualifying subgraph sets.
We evaluate the framework on SemGraph-Bench's 450 queries.

## Framework

The Coordinator interprets the query and selects investigations. Inspectors judge
the relevant source text in independent contexts. The runtime executes graph
operations, shares compatible evidence, and preserves original relation records.

## Pipeline

Query interpretation → structural search → semantic inspection and evidence
sharing → complete answer-set assembly.

## Performance

Results reported in the paper on all 450 SemGraph-Bench queries.

| Model | Accuracy (%) | F1 (%) |
| --- | ---: | ---: |
| GPT 5.6 Luna | 72.67 | 76.19 |
| DeepSeek V4 Pro | 75.11 | 76.81 |
| Gemini 3.8 Flash | 97.11 | 98.54 |

## Installation Steps

### 1. Create a Conda environment

```bash
conda create -n semgraph python=3.11
conda activate semgraph
```

### 2. Install dependencies

Run all commands from the repository root:

```bash
pip install -r requirements.txt
```

### 3. Download source graphs

The 450 questions and reference answers are included in `benchmarks/`.
Download the seven source graph files from [TEG-Datasets](https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/tree/main):

| Graph | File path in TEG-Datasets |
| --- | --- |
| Citation | `arxiv/processed/arxiv.pkl` |
| Amazon Movie | `amazon_movie/processed/movie.pkl` |
| Goodreads Children | `goodreads_children/processed/children.pkl` |
| Goodreads Crime | `goodreads_crime/processed/crime.pkl` |
| Goodreads History | `goodreads_history/processed/history.pkl` |
| Reddit | `reddit/processed/reddit.pkl` |
| Twitter | `twitter/processed/twitter.pkl` |

Pass the relevant downloaded `.pkl` file to `framework/run.py` with `--graph`.
