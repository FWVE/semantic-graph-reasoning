"""Static and Greedy investigation policies from the paper, Sections 6.6 and C.2."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel

from semantic_graph_reasoning.agentic.graph import Graph
from semantic_graph_reasoning.agentic.models import ExecutionAction, ExecutionLimits
from semantic_graph_reasoning.agentic.provider import Provider
from semantic_graph_reasoning.agentic.service import Solver
from semantic_graph_reasoning.agentic.workspace import Workspace

T = TypeVar("T", bound=BaseModel)
Strategy = Literal["static", "greedy"]


def identifier_key(value: str) -> tuple[tuple[int, str | int], ...]:
    """Compare numeric ID components numerically; clause names lexicographically."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", value))


def static_order(workspace: Workspace) -> list[str]:
    """Freeze the candidate-ID / clause-ID order of the initial unit domain."""
    def key(unit_id: str) -> tuple[Any, ...]:
        atoms = [workspace.atoms[a] for a in workspace.units[unit_id]["atom_ids"]]
        dependencies = sorted((identifier_key(c), str(a["condition_id"]))
                              for a in atoms for c in a["candidate_ids"])
        return tuple(dependencies), identifier_key(unit_id)

    return sorted(workspace.units, key=key)


def available_units(workspace: Workspace) -> list[str]:
    live = set(workspace.state()["live_atoms"])
    return [uid for uid, unit in workspace.units.items()
            if live.intersection(unit["atom_ids"])
            and (uid not in workspace.receipts or uid in workspace.blocked)
            and not (uid in workspace.blocked and uid in workspace.clarifications)]


def greedy_order(workspace: Workspace) -> list[str]:
    """Recompute distinct unresolved-candidate coverage at every stage.

    A compatible unit can hold several questions: its coverage is their union.
    Object IDs and clause IDs break ties, followed by the stable unit ID.
    """
    state = workspace.state()
    live = set(state["live_atoms"])
    unresolved = set(state["possible"]) - set(state["certain"])

    def key(unit_id: str) -> tuple[Any, ...]:
        unit = workspace.units[unit_id]
        atoms = [workspace.atoms[a] for a in unit["atom_ids"] if a in live]
        candidates = {c for atom in atoms for c in atom["candidate_ids"]} & unresolved
        objects = tuple(sorted(identifier_key(obj) for obj in unit["subjects"]))
        clauses = tuple(sorted(str(atom["condition_id"]) for atom in atoms))
        return -len(candidates), objects, clauses, identifier_key(unit_id)

    return sorted(available_units(workspace), key=key)


class SchedulingProvider:
    """Replace scheduling only; planning and every Inspector use the same provider."""

    def __init__(self, native: Provider, solver: ScheduledSolver) -> None:
        self.native, self.solver = native, solver
        self.initial_order: list[str] | None = None

    @property
    def calls(self) -> int:
        return self.native.calls

    @calls.setter
    def calls(self, value: int) -> None:
        self.native.calls = value

    @property
    def tokens(self) -> int:
        return self.native.tokens

    @tokens.setter
    def tokens(self, value: int) -> None:
        self.native.tokens = value

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                      output_type: type[T]) -> T:
        solver, workspace = self.solver, self.solver.workspace
        if role != "coordinator" or workspace is None:
            return await self.native.request(role, instructions, messages, output_type)
        if self.initial_order is None:
            self.initial_order = static_order(workspace)
        available = set(available_units(workspace))
        order = (greedy_order(workspace) if solver.strategy == "greedy"
                 else [uid for uid in self.initial_order if uid in available])
        solver.context_routes[-1] = {"role": "scheduler", "call": None,
                                     "modules": [solver.strategy]}
        if not order:
            return cast(T, ExecutionAction(action="finish", reason="No resolvable unit remains."))
        if order[0] in workspace.blocked:
            # Preserve the core's single clarification and source-reading contract.
            chosen = order[:1]
            reason = ("Re-read the specified source fields under the existing condition and "
                      "role bindings. Address the previous evidence issue; do not infer "
                      "unsupported facts or change the query interpretation.")
        else:
            chosen = []
            for uid in order:
                if uid in workspace.blocked or len(chosen) == 4:
                    break
                chosen.append(uid)
            reason = f"Execute the {solver.strategy} investigation order."
        solver.events.append({"role": "scheduler", "strategy": solver.strategy,
                              "unit_ids": chosen})
        return cast(T, ExecutionAction(action="inspect", unit_ids=chosen, reason=reason))


class ScheduledSolver(Solver):
    strategy: Strategy

    def __init__(self, graph: Graph, provider: Provider, limits: ExecutionLimits,
                 output_dir: Path | None = None) -> None:
        super().__init__(graph, provider, limits, output_dir)
        self.provider = SchedulingProvider(provider, self)


class StaticSolver(ScheduledSolver):
    runtime_name = "static"
    strategy: Strategy = "static"


class GreedySolver(ScheduledSolver):
    runtime_name = "greedy"
    strategy: Strategy = "greedy"
