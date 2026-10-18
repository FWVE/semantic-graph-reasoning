# Semantic Graph Reasoning

A Coordinator–Inspector framework for reasoning over text-rich graphs, with
**SemGraph-Bench: 450 queries across seven graphs and nine structural families**.

## Abstract

Answering a structural query over a text-rich graph requires both graph computation
and interpretation of node and edge text. Our framework uses a persistent
Coordinator to interpret the question, configure exact graph operations, and
choose semantic investigations. Each investigation uses a fresh Inspector context.
The runtime shares compatible judgments, tracks unresolved dependencies, and
returns the complete set of qualifying structures with their original edge records.
SemGraph-Bench evaluates these complete answers rather than a single example.

## Framework

The graph interface provides `pattern`, `shortest_paths`, and `anchored_core`.
The Coordinator selects and configures these operations from the public question.
Inspectors read original source text; the runtime checks evidence references,
shares judgments with identical dependencies, and preserves relation provenance.

## Pipeline

The solver receives the question, public anchors, and source graph interface.
Reference answers and frozen benchmark evidence are used only by the offline
evaluator. A run completes when the enumerated result set is resolved and all
output obligations are satisfied; unfinished runs return `unresolved`.

## Performance

| Model | Reasoning effort | Exact match | Completed |
| --- | --- | ---: | ---: |
| DeepSeek V4 Pro | none | 338/450 (75.11%) | 436/450 |
| GPT 5.6 Luna | none | 323/450 (71.78%) | 432/450 |
| Gemini 3.8 Flash | low | 433/450 (96.22%) | 450/450 |

These are recorded main-framework experiments with their respective source
snapshots. [Result metadata](framework/results/main450.json) and the accompanying
per-query CSVs preserve the measurement protocol and resource accounting.
Their native protocol compares anchor, node and original-edge sets; the evaluator
provided here also preserves ordered-path identity. The packaged source has been
checked locally and has not been rerun against model APIs.

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

### 3. Prepare SemGraph-Bench

The benchmark includes 450 questions, 6,683 structural candidates, 1,844 reference
answer structures, and 10 questions with empty reference answers. Each of the nine
structural families contains 50 questions.

| File | Content |
| --- | --- |
| [queries.csv](benchmarks/data/queries.csv) | Public input: `query_id`, `graph_domain`, `query`, `anchor_node_ids` |
| [gold.json](benchmarks/data/gold.json) | Complete reference answers |
| [sources.json](benchmarks/sources.json) | Source graph revisions, paths and checksums |
| [semgraph_bench_450.zip](benchmarks/data/semgraph_bench_450.zip) | Frozen resources and annotation evidence |

```bash
python benchmarks/verify.py
python benchmarks/download.py
python benchmarks/download.py --domains twitter --download
python benchmarks/download.py --domains twitter --verify
```

Omit `--domains` to prepare all seven source graphs (6.18 GB). The downloader uses
the pinned upstream revision and verifies file sizes and SHA256 values.

The sources are Citation, Reddit, Twitter, Goodreads Children, Goodreads History,
Goodreads Crime, and Amazon Movie from
[TEG-DB](https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets).
Source text and excerpts retain their upstream terms; the software license does
not relicense those datasets. Original annotation and audit records are included
with the frozen evidence.

### 4. Run the framework

Set the selected provider credential in your environment or a local `.env` file:
`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENROUTER_API_KEY`. Optional provider
endpoints are `OPENAI_BASE_URL` and `DEEPSEEK_BASE_URL`.

```bash
python framework/run.py preflight
python framework/run.py init --campaign framework/results/example --query-execution-limit 1 --acceptance-queries 1 --model gpt-5.6-terra --reasoning-effort high
python framework/run.py run --campaign framework/results/example --query-ids Q0001 --phase acceptance --allow-api
python framework/run.py status --campaign framework/results/example
```

Download the graph named in each selected question before running it.
`preflight` checks public inputs without loading reference answers or calling a
model. `run --allow-api` executes the selected model. Use `--help` for model and
budget options. Model settings and execution limits are fixed per campaign;
failed attempts also consume an execution slot.

### 5. Evaluate

```bash
python benchmarks/evaluate.py --campaign framework/results/example --phase acceptance --query-ids Q0001 --output framework/results/example-score
```

Omit `--query-ids` to score all 450 questions. The evaluator selects the latest
attempt for each question within the specified phase and freezes predictions
before opening references. It reports complete answer-set exact match,
precision/recall/F1, and invalid or duplicate answers. Missing and unresolved
predictions count as incorrect, including on empty-reference questions.

For external systems, use `--predictions predictions.json` instead of `--campaign`.
The JSON maps each query ID to `{"status": "complete", "answers": [...]}`. Every
answer includes integer `anchor_node_id`, `node_ids`, and original `edge_ids`; path answers
also include ordered `path_node_ids`. A complete empty answer uses `"answers": []`.
