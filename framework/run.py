"""Run training-free semantic graph agents against the public 450-query CSV."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from semantic_graph_reasoning.agentic.batch import BatchGuard
from semantic_graph_reasoning.agentic.campaign import complete, reserve
from semantic_graph_reasoning.agentic.graph import Graph
from semantic_graph_reasoning.agentic.models import ExecutionLimits
from semantic_graph_reasoning.agentic.provider import (
    SUPPORTED_MODELS,
    NativeProvider,
    validate_model_effort,
    write_json,
)
from semantic_graph_reasoning.agentic.service import Solver
from semantic_graph_reasoning.agentic.workspace import digest
from semantic_graph_reasoning.data.benchmark import PUBLIC_CSV, SOURCES, load_graph, public_rows
from semantic_graph_reasoning.scheduling import GreedySolver, StaticSolver


def fingerprints() -> dict[str, str]:
    package = ROOT / "framework/semantic_graph_reasoning"
    paths = sorted(p for p in package.rglob("*.py") if p.relative_to(package).parts[0] != "experiments")
    paths += sorted((ROOT / "framework").glob("*.py"))
    paths += [PUBLIC_CSV, SOURCES, ROOT / "requirements.txt"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _identity_graph(graph: Graph) -> Graph:
    return graph


@dataclass(frozen=True)
class Runtime:
    """Explicit entry-point dependencies; no experiment discovery or default routing."""

    solver: type[Solver] = Solver
    limits: type[ExecutionLimits] = ExecutionLimits
    graph_adapter: Callable[[Graph], Graph] = _identity_graph
    source_files: tuple[Path, ...] = ()

    @property
    def name(self) -> str:
        return self.solver.runtime_name

    def fingerprints(self) -> dict[str, str]:
        extra = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source_files}
        return fingerprints() | extra


CORE = Runtime()


def initialize(args: argparse.Namespace, *, runtime: Runtime = CORE) -> dict[str, Any]:
    if not 1 <= args.acceptance_queries <= args.query_execution_limit:
        raise ValueError("acceptance count must be between one and the query cap")
    model, effort = args.model or "deepseek-v4-pro", args.reasoning_effort or "none"
    validate_model_effort(model, effort)
    args.campaign.mkdir(parents=True, exist_ok=False)
    protected = [PUBLIC_CSV, ROOT / "benchmarks/data/gold.json", SOURCES]
    value = {"model": model, "reasoning_effort": effort, "runtime": runtime.name,
             "query_execution_limit": args.query_execution_limit, "acceptance_queries": args.acceptance_queries,
             "authorization_note": "Run only with explicit user API authorization in the current conversation.",
             "protected": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}}
    write_json(args.campaign / "authorization.json", value)
    return {"campaign_created": str(args.campaign), "external_call_performed": False, **value}


async def run(args: argparse.Namespace, *, guard: BatchGuard | None = None, runtime: Runtime = CORE) -> dict[str, Any]:
    if not args.allow_api:
        raise ValueError("run requires --allow-api and current user authorization")
    guard = guard or BatchGuard()
    root = args.campaign.resolve()
    authorization = json.loads((root / "authorization.json").read_text(encoding="utf-8"))
    if authorization.get("runtime") != runtime.name:
        raise ValueError("campaign runtime differs from this entry point; keep core and experiments separate")
    effort = authorization["reasoning_effort"]
    validate_model_effort(authorization["model"], effort)
    if args.model is not None and args.model != authorization["model"]:
        raise ValueError("run cannot override the campaign's authorized model")
    if getattr(args, "reasoning_effort", None) not in (None, effort):
        raise ValueError("run cannot override the campaign's authorized reasoning effort")
    rows = public_rows(args.queries_csv)
    if args.queries_csv.resolve() != PUBLIC_CSV.resolve():
        raise ValueError("this 450-query campaign uses the canonical public CSV; custom inputs need a separate adapter")
    ids = args.query_ids.split(",")
    if len(set(ids)) != len(ids) or not set(ids) <= rows.keys():
        raise ValueError("select explicit distinct public QIDs")
    limits = runtime.limits()
    frozen = runtime.fingerprints()
    source_hash = digest(frozen)
    if args.phase == "acceptance":
        freeze_path = root / "acceptance_freeze.json"
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if freeze["source_hash"] != source_hash or freeze.get("runtime") != runtime.name:
                raise ValueError("source changed after acceptance freeze; freeze a new acceptance version explicitly")
            if (freeze["model"] != authorization["model"] or freeze["reasoning_effort"] != effort
                    or freeze["limits"] != limits.model_dump(mode="json")):
                raise ValueError("model or limits changed after acceptance freeze")
        else:
            write_json(freeze_path, {"source_hash": source_hash, "files": frozen,
                                     "runtime": runtime.name,
                                     "model": authorization["model"], "reasoning_effort": effort,
                                     "limits": limits.model_dump(mode="json")})
    cache: dict[str, Any] = {}
    results = []
    for query_id in ids:
        if guard.failure is not None:
            break
        if runtime.fingerprints() != frozen:
            raise ValueError("source changed during this batch; not launching another query")
        row = rows[query_id]
        domain = row["graph_domain"]
        if domain not in cache:
            print(json.dumps({"loading_graph": domain}), flush=True)
            loaded, schema = load_graph(domain)
            cache[domain] = runtime.graph_adapter(loaded), schema
        graph, schema = cache[domain]
        # Every attempt, including failures and reruns, reserves a persistent slot first.
        index, output = reserve(root, query_id, args.phase, source_hash)
        write_json(output / "run_configuration.json", {"query_id": query_id, "phase": args.phase,
                   "runtime": runtime.name,
                   "model": authorization["model"], "reasoning_effort": effort, "source_hash": source_hash,
                   "files": frozen, "limits": limits.model_dump(mode="json"), "source": schema})
        print(json.dumps({"started_run": index, "query_id": query_id, "limit": authorization["query_execution_limit"]}), flush=True)
        provider = NativeProvider(authorization["model"], limits, output, guard=guard, reasoning_effort=effort)
        try:
            solver = runtime.solver(graph, provider, limits, output)
            result = await solver.run(row["query"], row["anchor_node_ids"], schema)
            complete(root, index, result)
            if result.get("failure_scope") == "batch" and guard.failure is None:
                guard.stop(RuntimeError(f"query {query_id} unresolved: {result.get('reason', '')}"))
                write_json(root / "batch_stop.json", {**(guard.failure or {}),
                           "trigger_run": output.name, "trigger_stage": "query_result",
                           "dispatch_stopped": True, "peer_cancellation_requested": True})
        finally:
            await provider.close()
        summary = {"run": index, "query_id": query_id, "status": result["status"],
                   "answers": len(result["answers"]) if result.get("answers") is not None else None,
                   "audit": result["audit"]}
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return {"runs": results}


def parser(commands: tuple[str, ...] = ("init", "preflight", "run", "status")) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=commands)
    value.add_argument("--strategy", choices=["full", "static", "greedy"], default="full")
    value.add_argument("--campaign", type=Path, default=ROOT / "artifacts/framework/work/semgraph_450")
    value.add_argument("--queries-csv", type=Path, default=PUBLIC_CSV)
    value.add_argument("--query-ids", default="")
    value.add_argument("--phase", choices=["debug", "acceptance"], default="debug")
    value.add_argument("--allow-api", action="store_true")
    value.add_argument("--model", choices=SUPPORTED_MODELS,
                       help="Model pinned at init (default: deepseek-v4-pro); run cannot override the campaign model.")
    value.add_argument("--reasoning-effort", choices=["high", "none", "low"],
                       help="Pinned at init (default: none); Gemini OpenRouter requires low; native models support high/none.")
    value.add_argument("--query-execution-limit", type=int, default=30)
    value.add_argument("--acceptance-queries", type=int, default=10)
    return value


def execute(args: argparse.Namespace, *, runtime: Runtime = CORE) -> dict[str, Any]:
    if args.command == "init":
        result = initialize(args, runtime=runtime)
    elif args.command == "run":
        result = asyncio.run(run(args, runtime=runtime))
    elif args.command == "status":
        path = args.campaign / "ledger.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    else:
        rows = public_rows(args.queries_csv)
        selected = args.query_ids.split(",") if args.query_ids else list(rows)
        if not set(selected) <= rows.keys():
            raise ValueError("unknown public QID")
        result = {"public_rows": len(rows), "selected_rows": len(selected),
                  "domains": sorted({rows[q]["graph_domain"] for q in selected}),
                  "private_inputs_loaded": False, "external_call_performed": False,
                  "source_hash": digest(runtime.fingerprints()), "runtime": runtime.name}
    return result


def main() -> None:
    args = parser().parse_args()
    solver = {"full": Solver, "static": StaticSolver, "greedy": GreedySolver}[args.strategy]
    result = execute(args, runtime=Runtime(solver=solver))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
