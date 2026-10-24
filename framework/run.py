"""Run one public question against its complete source graph."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from semantic_graph_reasoning.data import TEXT_SEMANTICS, load_graph, public_rows
from semantic_graph_reasoning.models import ExecutionLimits
from semantic_graph_reasoning.provider import OPENROUTER_MODEL, SUPPORTED_MODELS, NativeProvider
from semantic_graph_reasoning.service import Solver

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    question = value.add_mutually_exclusive_group(required=True)
    question.add_argument("--query-id", help="Question ID in benchmarks/queries.csv")
    question.add_argument("--query", help="A natural-language graph question")
    value.add_argument("--questions", type=Path, default=ROOT / "benchmarks/queries.csv")
    value.add_argument("--graph", type=Path, required=True, help="Complete original source graph (.pkl)")
    value.add_argument("--domain", choices=tuple(TEXT_SEMANTICS), help="Required with --query")
    value.add_argument("--anchors", type=int, nargs="+", help="Public anchor node IDs, required with --query")
    value.add_argument("--model", choices=SUPPORTED_MODELS, default="deepseek-v4-pro")
    value.add_argument("--reasoning-effort", choices=("none", "high", "low"))
    value.add_argument("--output", type=Path, help="Save the final answer JSON")
    return value


def question_input(args: argparse.Namespace) -> tuple[str, list[int], str]:
    if args.query_id:
        if args.domain is not None or args.anchors is not None:
            raise ValueError("--query-id uses the question's published graph and anchor IDs")
        rows = public_rows(args.questions.resolve())
        if args.query_id not in rows:
            raise ValueError(f"unknown question: {args.query_id}")
        row = rows[args.query_id]
        return str(row["query"]), row["anchor_node_ids"], str(row["graph_domain"])
    if args.domain is None or not args.anchors:
        raise ValueError("--query requires --domain and --anchors")
    return str(args.query), args.anchors, str(args.domain)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    query, anchors, domain = question_input(args)
    graph, schema = load_graph(domain, args.graph.resolve())
    limits = ExecutionLimits()
    effort = args.reasoning_effort or ("low" if args.model == OPENROUTER_MODEL else "none")
    provider = NativeProvider(args.model, limits, reasoning_effort=effort)
    try:
        return await Solver(graph, provider, limits).run(query, anchors, schema)
    finally:
        await provider.close()


def main() -> None:
    args = parser().parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("choose a new output file")
    result = asyncio.run(run(args))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, "utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
