"""450-row PUBLIC input adapter and existing audited full-graph loader composition."""
from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

from semantic_graph_reasoning.agentic.graph import Graph
from semantic_graph_reasoning.data.adapters import prepare_framework_dataset

ROOT = Path(__file__).resolve().parents[3]
PUBLIC_CSV = ROOT / "benchmarks/data/queries.csv"
SOURCES = ROOT / "benchmarks/sources.json"


def review_author_aliases(sources: Iterable[int], texts: Iterable[str]) -> dict[int, tuple[str, ...]]:
    """Read identity metadata from the native review header, never from review-body mentions."""
    header = re.compile(r"\AReviewer (.+?) left a review on \d{1,2} \d{1,2}, \d{4}, giving the product ")
    names: dict[int, set[str]] = {}
    for source, text in zip(sources, texts, strict=True):
        match = header.match(text)
        if match:
            names.setdefault(int(source), set()).add(match.group(1))
    return {node: tuple(sorted(aliases)) for node, aliases in names.items()}


def public_rows(path: Path = PUBLIC_CSV) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"query_id", "graph_domain", "query", "anchor_node_ids"}:
            raise ValueError("framework input must contain exactly the four public CSV columns")
        rows = list(reader)
    result = {}
    for row in rows:
        anchors = json.loads(row["anchor_node_ids"])
        if not isinstance(anchors, list) or not anchors or any(type(n) is not int for n in anchors):
            raise ValueError("invalid public anchor IDs")
        if not row["query"].strip() or row["query_id"] in result:
            raise ValueError("empty question or duplicate public query ID")
        result[row["query_id"]] = {**row, "anchor_node_ids": anchors}
    return result


def load_graph(domain: str) -> tuple[Graph, dict[str, Any]]:
    registry = json.loads(SOURCES.read_text(encoding="utf-8"))
    spec = next(s for s in registry["graph_sources"] if s["graph_instance"] == domain)
    source = (ROOT / spec["path"]).resolve()
    if not source.is_relative_to(ROOT / "benchmarks/sources"):
        raise ValueError("registered graph path escaped benchmarks/sources")
    if source.stat().st_size != spec["size_bytes"]:
        raise ValueError("raw graph size differs from the registered source")
    dataset = prepare_framework_dataset(domain, source)
    source_hash = str(dataset.source_manifest["sha256"])
    compact = dataset.compact_graph
    if compact.weighted:
        raise ValueError("this runtime exposes unweighted operations only")
    fields = dataset.semantic_materialization_schema or {"node": ["text", "title"], "edge": ["text"]}
    # This source stores actor nodes as "reviewer"; public actor names live in
    # the mechanically formatted header of their source-incident review records.
    identity_aliases = (review_author_aliases(compact.edge_sources, dataset.content_provider.mapping["text_edges"])
                        if domain == "amazon_movie" else {})

    def read(kind: str, identifier: int, names: list[str]) -> dict[str, Any]:
        return dict(dataset.content_provider.materialize(entity_type=kind, entity_id=str(identifier), permitted_fields=names))

    @cache
    def kind(node: int) -> str:
        return str(read("node", node, ["node_kind"])["node_kind"]) if "node_kind" in fields["node"] else "paper"

    def name(node: int) -> str:
        if node in identity_aliases:
            return "; ".join(identity_aliases[node])
        field = "name" if "name" in fields["node"] else "title" if "title" in fields["node"] else None
        return str(read("node", node, [field])[field]) if field else f"{kind(node)} {node}"

    graph = Graph(node_count=compact.node_count, sources=compact.edge_sources, targets=compact.edge_targets,
                  text=lambda entity_kind, identifier: str(read(entity_kind, identifier, ["text"])["text"]),
                  kind=kind, name=name, domain=domain, source_hash=source_hash,
                  original_edge_ids=dataset.external_edge_ids,
                  node_kinds=tuple(sorted(spec["text_field_provenance"]["node_text_by_kind"])))
    schema = {"domain": domain, "directed": compact.directed, "weighted": False,
              "node_count": compact.node_count, "edge_count": compact.edge_count,
              "node_kinds": sorted(spec["text_field_provenance"]["node_text_by_kind"]),
              "text_fields": ["text"], "parallel_edges": "distinct original records",
              "text_semantics": {"node": {kind: field["semantic_eligibility"]
                                          for kind, field in spec["text_field_provenance"]["node_text_by_kind"].items()},
                                 "edge": spec["text_field_provenance"]["edge_text"]["semantic_eligibility"]},
              "edge_direction": "stored source to target; original ID restored only at output",
              "identity_lookup": ("Native node names/titles; Amazon reviewer aliases come only from source-bound native review headers."
                                  if identity_aliases else "Native node names/titles and declared typed IDs."),
              "text_projection": spec["text_field_provenance"].get("projection_summary")}
    return graph, schema
