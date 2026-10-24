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

| Paper component | Implementation |
| --- | --- |
| Coordinator and `CoordinatorPlan` | [`coordinator_prompt`](framework/semantic_graph_reasoning/agentic/prompts.py#L221), with the planning and investigation modules |
| Inspector | [`inspector_prompt`](framework/semantic_graph_reasoning/agentic/prompts.py#L278), including source reading, clarification, and evidence quotation |
| Blueprint and scheduling schema | [`Plan`](framework/semantic_graph_reasoning/agentic/models.py#L86), [`Action` and `ExecutionAction`](framework/semantic_graph_reasoning/agentic/models.py#L109) |
| Runtime inputs and evidence feedback | [`context.py`](framework/semantic_graph_reasoning/agentic/context.py) and [`service.py`](framework/semantic_graph_reasoning/agentic/service.py) |
| Static / Greedy scheduling (Table 4, Appendix C.2) | [`static_order`](framework/semantic_graph_reasoning/scheduling.py#L26) and [`greedy_order`](framework/semantic_graph_reasoning/scheduling.py#L45) |

Static fixes the initial unit order by candidate ID and clause ID. Greedy recomputes
coverage of distinct unresolved candidates at each stage, breaking ties by object
ID and clause ID. Both use the shared Inspector, judgment cache, short-circuit
propagation, evidence follow-up, and a maximum of four units per stage.

## Pipeline

The solver receives the question, public anchors, and source graph interface.
Reference answers and frozen benchmark evidence are used only by the offline
evaluator. A run completes when the enumerated result set is resolved and all
output obligations are satisfied; unfinished runs return `unresolved`.

[`framework/run.py`](framework/run.py) selects `--strategy full`, `static`, or
`greedy`. The shared per-query limits are **300 seconds, 160 model requests, and
400,000 tokens**. Model temperature is 0; reasoning is disabled for DeepSeek Pro
and Luna, and set to low for Gemini Flash. Input validation requires no API calls:

```bash
python framework/run.py preflight --strategy full
```

## Performance

| Model | Reasoning effort | Exact match (Accuracy) | F1 (%) |
| --- | --- | ---: | ---: |
| DeepSeek V4 Pro | none | 338/450 (75.11%) | 76.81 |
| GPT 5.6 Luna | none | 327/450 (72.67%) | 76.19 |
| Gemini 3.8 Flash | low | 437/450 (97.11%) | 98.54 |

Recorded results and scoring protocols are available in
[framework/results](framework/results), with a [summary](framework/results/summary.json)
and per-query CSVs for each model. The `em` column sums to the exact-match counts
in Tables 3 and A4. Resource summaries correspond to Table A6.

[`benchmarks/evaluate.py`](benchmarks/evaluate.py) scores sets after `canon_q`
normalization: equivalent repeated answer groups count once. Original node and
relation-record IDs, anchor bindings, parallel records, and directed path order
remain part of instance identity. Accuracy is complete answer-set equality; F1
is averaged equally over all 450 queries. Explicit empty/empty answers receive
F1 = 1.00; missing, unfinished, or invalid outputs receive F1 = 0.00.

The CSVs retain integer instance counts: `tp`, `predicted_groups`, and
`reference_groups`. Reconstruct each row's F1 as
`2 * tp / (predicted_groups + reference_groups)`; when both counts are zero, use
`em`. Average these fractions over all 450 rows, then multiply by 100 and round
to two decimal places to reproduce the F1 percentages above.

Appendix C.2 reports **23 unfinished DeepSeek queries under the 300-second or
160-request budget cap**.

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
