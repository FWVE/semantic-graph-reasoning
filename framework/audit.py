"""Replay real API transcripts against public inputs and raw graphs, without API calls."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))
sys.path.insert(0, str(ROOT / "framework"))

from pydantic import BaseModel
from run import CORE, Runtime

from masge.agentic.deepseek_strict import ENDPOINT, parse_tool_output, strict_request
from masge.agentic.models import ExecutionAction, ExecutionLimits
from masge.agentic.openrouter import ENDPOINT as OPENROUTER_ENDPOINT
from masge.agentic.openrouter import MODEL as OPENROUTER_MODEL
from masge.agentic.openrouter import parse_structured_output, structured_request, validate_response
from masge.agentic.provider import (
    DEEPSEEK_MODELS,
    parse_output,
    request_input,
    validate_model_effort,
    write_json,
)
from masge.agentic.workspace import digest
from masge.data.benchmark import load_graph, public_rows

T = TypeVar("T", bound=BaseModel)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ReplayProvider:
    def __init__(self, path: Path, model: str, reasoning_effort: str = "high",
                 limits: ExecutionLimits | None = None) -> None:
        validate_model_effort(model, reasoning_effort)
        self.directories = sorted((path / "calls").iterdir())
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.calls = self.tokens = 0
        self.limits = limits or ExecutionLimits()

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                      output_type: type[T]) -> T:
        require(self.calls < len(self.directories), "replay requested an unrecorded provider call")
        directory = self.directories[self.calls]
        request, response, accounting = (read(directory / name) for name in
                                         ("request.json", "response.json", "accounting.json"))
        require(directory.name.endswith("_" + role), "provider role/order mismatch")
        require(request["model"] == response["model"] == self.model, "actual native model mismatch")
        require(request["reasoning_effort"] == self.reasoning_effort and request["store"] is False, "request settings mismatch")
        deepseek = self.model in DEEPSEEK_MODELS
        openrouter = self.model == OPENROUTER_MODEL
        if openrouter:
            validate_response(response)
        if not deepseek and not openrouter:
            require(response["reasoning"]["effort"] == self.reasoning_effort, "response reasoning effort mismatch")
        require(accounting["model"] == self.model and accounting["reasoning_effort"] == self.reasoning_effort,
                "accounting model/effort mismatch")
        if self.reasoning_effort == "none":
            details = "completion_tokens_details" if deepseek else "output_tokens_details"
            reasoning_output = (any(c.get("message", {}).get("reasoning_content") for c in response.get("choices", []))
                                if deepseek else any(item.get("type") == "reasoning" for item in response["output"]))
            reasoning_tokens = ((response.get("usage") or {}).get(details) or {}).get("reasoning_tokens")
            valid_counter = reasoning_tokens in (None, 0) if deepseek else reasoning_tokens == 0
            require(valid_counter and not reasoning_output,
                    "non-thinking response contains reasoning or incompatible usage")
        require(accounting["status"] == "ok", "provider failure")
        if not deepseek and not openrouter:
            require(response["status"] == "completed", "provider failure")
        expected_format = ("function_tool" if issubclass(output_type, ExecutionAction) else "strict_tool") if deepseek else "json_schema"
        require(request["output_format"] == expected_format, "provider output format mismatch")
        require(request["input"] == request_input(instructions, messages),
                "recorded model input differs from public-only replay")
        require(request["output_schema"] == output_type.model_json_schema(), "output schema mismatch")
        if openrouter:
            wire = structured_request(self.model, self.reasoning_effort, request["input"],
                                      output_type, self.limits.max_output_tokens)
            require(request["endpoint"] == OPENROUTER_ENDPOINT and request["provider_request"] == wire,
                    "pinned OpenRouter request/history differs from replay")
            require(accounting["actual_provider"] == response["provider"]
                    and accounting["actual_service_tier"] == response["service_tier"], "routing accounting mismatch")
            parsed = parse_structured_output(response, output_type)
        elif deepseek:
            wire = strict_request(self.model, self.reasoning_effort, request["input"],
                                  output_type, self.limits.max_output_tokens)
            require(request["endpoint"] == ENDPOINT and request["provider_request"] == wire,
                    "strict provider request/history differs from replay")
            parsed = parse_tool_output(response, output_type)
        else:
            parsed = parse_output(response, output_type)
        require(response["usage"] == accounting["usage"], "provider accounting mismatch")
        self.calls += 1
        self.tokens += response["usage"]["total_tokens"]
        return parsed


async def audit(campaign: Path, phase: str, *, runtime: Runtime = CORE) -> dict[str, Any]:
    authorization, ledger = read(campaign / "authorization.json"), read(campaign / "ledger.json")
    require(authorization.get("runtime") == runtime.name, "campaign runtime differs from this replay entry point")
    validate_model_effort(authorization["model"], authorization["reasoning_effort"])
    require(len(ledger["runs"]) <= authorization["query_execution_limit"], "authorized query cap exceeded")
    frozen = runtime.fingerprints()
    if phase == "acceptance":
        freeze = read(campaign / "acceptance_freeze.json")
        require(freeze.get("runtime") == runtime.name, "acceptance freeze runtime mismatch")
        require(freeze["files"] == frozen, "current source differs from freeze")
        require(freeze["model"] == authorization["model"] and freeze["reasoning_effort"] == authorization["reasoning_effort"],
                "authorized model/effort differs from freeze")
    rows, graphs, results = public_rows(), {}, []
    for entry in ledger["runs"]:
        if entry["phase"] != phase:
            continue
        path = campaign / entry["path"]
        try:
            config, recorded = read(path / "run_configuration.json"), read(path / "result.json")
            require(config.get("runtime") == recorded.get("runtime") == runtime.name, "run/result runtime mismatch")
            require(config["files"] == frozen and entry["source_hash"] == digest(frozen), "run uses an older source version")
            require(config["model"] == authorization["model"] and config["reasoning_effort"] == authorization["reasoning_effort"],
                    "run model/effort differs from authorization")
            require(recorded["audit"]["handoff_eligible"], "recorded run contains an anomaly or is unresolved")
            row = rows[entry["query_id"]]
            if row["graph_domain"] not in graphs:
                loaded, schema = load_graph(row["graph_domain"])
                graphs[row["graph_domain"]] = runtime.graph_adapter(loaded), schema
            graph, schema = graphs[row["graph_domain"]]
            require(config["source"] == schema, "raw source/schema mismatch")
            public = read(path / "public_input.json")
            require(public["query"] == row["query"] and public["public_anchor_node_ids"] == row["anchor_node_ids"],
                    "query/anchor input differs from public CSV")
            require(public["graph_schema"] == schema and public["limits"] == config["limits"], "public schema/limits mismatch")
            provider = ReplayProvider(path, authorization["model"], authorization["reasoning_effort"],
                                      runtime.limits.model_validate(config["limits"]))
            solver = runtime.solver(graph, provider, runtime.limits.model_validate(config["limits"]))
            result = await solver.run(row["query"], row["anchor_node_ids"], schema)
            result["audit"].pop("seconds")
            recorded["audit"].pop("seconds")
            require(result == recorded, "replayed result/audit differs")
            require(provider.calls == len(provider.directories), "unused recorded calls")
            require(digest(solver.events) == digest(read(path / "events.json")), "action/evidence/feedback transcript mismatch")
            require(solver.memory_snapshot() == read(path / "reasoning_memory.json"), "execution memory differs from replay")
            require(solver.workspace is not None and digest(solver.workspace.snapshot()) == digest(read(path / "workspace.json")),
                    "grounding, propagation, receipts or output differ")
            usage = read(path / "provider_usage.json")
            require(usage["calls"] == provider.calls and usage["tokens"] == provider.tokens, "total accounting mismatch")
            results.append({"run": entry["run"], "query_id": entry["query_id"], "passed": True,
                            "answers": len(result["answers"]), "calls": provider.calls, "tokens": provider.tokens})
        except Exception as error:
            results.append({"run": entry["run"], "query_id": entry["query_id"], "passed": False,
                            "error": f"{type(error).__name__}: {error}"})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    protected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
                 for name, expected in authorization["protected"].items()}
    passed = [r for r in results if r["passed"]]
    selected = {}
    for row in passed:
        selected.setdefault(row["query_id"], row["run"])
    report = {"phase": phase, "runtime": runtime.name, "source_hash": digest(frozen), "query_executions_used": len(ledger["runs"]),
              "query_execution_limit": authorization["query_execution_limit"], "results": results,
              "protected_files_unchanged": protected, "external_calls_performed": False,
              "selected_successful_runs": dict(list(selected.items())[:authorization["acceptance_queries"]]),
              "handoff_ready": phase == "acceptance" and all(protected.values())
                  and len(selected) >= authorization["acceptance_queries"],
              "scope": "Public-only transcript replay, exact source grounding and runtime invariants; not semantic accuracy."}
    write_json(campaign / f"{phase}_replay_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "artifacts/framework/work/agentic_450_terra_high")
    parser.add_argument("--phase", choices=["debug", "acceptance"], default="acceptance")
    args = parser.parse_args()
    result = asyncio.run(audit(args.campaign, args.phase))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["handoff_ready"] or args.phase == "debug" else 1)
