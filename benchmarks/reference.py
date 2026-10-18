"""Frozen benchmark structures for OFFLINE checks only; never import from a solver."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/data"
ARCHIVE = OUTPUT / "semgraph_bench_450.zip"


def key(anchor: int, nodes: list[int], edges: list[int], path: list[int] | None = None) -> tuple:
    return anchor, tuple(sorted(set(nodes))), tuple(sorted(set(edges))), tuple(path) if path is not None else None


def reference(resource: dict, certificate: dict) -> tuple[set, set, bool]:
    """Preserve original relation records, public anchors, and ordered path identity."""
    reader, aliases = resource["reader"], resource["aliases"]
    topology = reader["topology"]
    role = topology["anchor_role"]
    is_path = topology["structural_family"] == "path"
    nodes, edges = aliases["node_alias_to_source_index"], aliases["edge_alias_to_source_index"]
    candidates = set()
    for mapping in reader["mappings"]:
        roles = mapping["role_node_aliases"]
        ordered = [nodes[roles[r]] for r in topology["role_order"]] if is_path else None
        candidates.add(key(nodes[roles[role]], [nodes[n] for n in roles.values()],
                           [edges[e] for relation in mapping["relations"] for e in relation["parallel_edge_aliases"]], ordered))
    expected = set()
    for group in certificate["complete_answer_groups"]:
        if group["record_kind"] == "cohesive_component":
            expected.add(key(group["anchor_node_id"], group["component_node_ids"],
                             [e["original_edge_id"] for e in group["all_internal_original_edge_records"]]))
        else:
            expected.add(key(group["role_node_ids"][role], group["supporting_node_ids"],
                             [e for relation in group["supporting_relations"] for e in relation["all_structural_original_edge_ids"]],
                             group["canonical_role_tuple"] if is_path else None))
    if len(candidates) != len(reader["mappings"]) or len(expected) != certificate["complete_answer_count"]:
        raise ValueError("reference normalization changed frozen structure counts")
    if not expected <= candidates:
        raise ValueError("reference answers escaped the frozen structural domain")
    return candidates, expected, is_path


def resource_from_archive(archive: zipfile.ZipFile, qid: str) -> dict:
    return json.loads(archive.read(f"benchmarks/resources/{qid}.json"))
