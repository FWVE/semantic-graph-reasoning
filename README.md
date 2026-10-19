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

Recorded results and scoring protocols are available in
[framework/results](framework/results), with a [summary](framework/results/summary.json)
and per-query CSVs for each model.

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
