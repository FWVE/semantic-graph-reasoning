"""Read complete source graphs while preserving their text and original edge IDs."""
from __future__ import annotations

import collections
import csv
import io
import json
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from semantic_graph_reasoning.graph import Graph


def public_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"query_id", "graph_domain", "query", "anchor_node_ids"}:
            raise ValueError("questions must contain only the four public input columns")
        result = {}
        for row in reader:
            anchors = json.loads(row["anchor_node_ids"])
            if not isinstance(anchors, list) or not anchors or any(type(n) is not int for n in anchors):
                raise ValueError("invalid public anchor IDs")
            if not row["query"].strip() or row["query_id"] in result:
                raise ValueError("empty question or duplicate query ID")
            result[row["query_id"]] = {**row, "anchor_node_ids": anchors}
    return result


def load_graph(domain: str, path: Path) -> tuple[Graph, dict[str, Any]]:
    with path.open("rb") as stream:
        source = RestrictedUnpickler(stream).load()
    if isinstance(source, (tuple, list)):
        if len(source) != 1:
            raise ValueError("expected one source graph")
        source = source[0]
    return graph_from_mapping(domain, cast(dict[str, Any], source._store._mapping))


def graph_from_mapping(domain: str, mapping: Mapping[str, Any]) -> tuple[Graph, dict[str, Any]]:
    """Use the entire source topology; no question-specific candidate filtering."""
    if domain not in TEXT_SEMANTICS:
        raise ValueError(f"unknown graph domain: {domain}")
    node_texts = ["" if text is None else str(text) for text in mapping["text_nodes"]]
    edge_texts = ["" if text is None else str(text) for text in mapping["text_edges"]]
    edges = np.asarray(mapping["edge_index"], dtype=np.int64)
    if edges.shape != (2, len(edge_texts)):
        raise ValueError("edge_index and text_edges must align one-for-one")
    sources, targets = edges
    original_ids = tuple(range(len(edge_texts)))
    names: dict[int, str] = {}
    kinds: dict[int, str] = {}
    identity_aliases: dict[int, set[str]] = {}
    if domain == "reddit":
        if len(node_texts) != 2 * len(edge_texts):
            raise ValueError("Reddit source must contain two endpoint texts per edge")
        for edge_id, (source, target) in enumerate(zip(sources, targets, strict=True)):
            community = node_texts[2 * edge_id].removeprefix("subreddit ").strip()
            user = node_texts[2 * edge_id + 1].removeprefix("user ").split(" has flair ", 1)[0].strip()
            for node, name, kind in [(int(source), user, "user"), (int(target), community, "subreddit")]:
                if node in names and (names[node] != name or kinds[node] != kind):
                    raise ValueError("conflicting Reddit endpoint identities")
                names[node], kinds[node] = name, kind
        if not names:
            raise ValueError("empty Reddit graph")
        node_texts = [f"{kinds[n]} {names[n]}" if n in names else "" for n in range(max(names) + 1)]
    elif domain == "twitter":
        projected_sources, projected_targets, original_ids, projected_texts = _twitter_author_projection(mapping)
        sources, targets = np.asarray(projected_sources), np.asarray(projected_targets)
        for node, original in enumerate(node_texts):
            kind = "tweet" if original == "tweet" else "user"
            kinds[node] = kind
            text = projected_texts[node]
            names[node] = (f"User {node}" if kind == "user" else
                           f"Tweet {node}: {text[:72]}" if text else f"Tweet {node}")
        node_texts = projected_texts
        edge_texts = [edge_texts[index] for index in original_ids]
    elif domain == "citation":
        if len(mapping["node_labels"]) != len(node_texts):
            raise ValueError("citation node labels and text fields disagree")
        title_pattern = re.compile(r"^The paper titled ['\"](?P<title>.+?)['\"] presents\s*", re.DOTALL)
        for node, text in enumerate(node_texts):
            match = title_pattern.match(text)
            names[node] = (" ".join(match.group("title").split()) if match else
                           " ".join(text.split())[:180] or "Untitled paper")
            kinds[node] = "paper"
    else:
        left, right = ("user", "item") if domain == "amazon_movie" else ("reader", "book")
        labels = mapping["text_node_labels"]
        if len(labels) != len(node_texts):
            raise ValueError("review node labels and text fields disagree")
        for node, text in enumerate(node_texts):
            is_left = labels[node] == -1 or str(labels[node]).strip() == "-1"
            kinds[node] = left if is_left else right
            if is_left:
                names[node] = f"{left.title()} {node}"
            elif domain == "amazon_movie":
                match = re.search(r"The product titled ['\"](.*?)['\"]\.", text, re.DOTALL)
                names[node] = match.group(1) if match else f"Item {node}"
            else:
                match = re.search(r"This book tittled (.*?) is a ", " ".join(text.split()), re.DOTALL)
                names[node] = " ".join(match.group(1).split()).strip() if match else f"Book node {node}"
        for source, target in zip(sources, targets, strict=True):
            if kinds[int(source)] != left or kinds[int(target)] != right:
                raise ValueError("review edges must connect an actor to a reviewed object")
        if domain == "amazon_movie":
            header = re.compile(r"\AReviewer (.+?) left a review on \d{1,2} \d{1,2}, \d{4}, giving the product ")
            for source, text in zip(sources, edge_texts, strict=True):
                match = header.match(text)
                if match:
                    identity_aliases.setdefault(int(source), set()).add(match.group(1))
            names.update({node: "; ".join(sorted(aliases)) for node, aliases in identity_aliases.items()})
    if any(int(node) < 0 or int(node) >= len(node_texts) for node in (*sources, *targets)):
        raise ValueError("source graph endpoint is outside the node range")
    spec = TEXT_SEMANTICS[domain]
    node_kinds = tuple(sorted(spec["node"]))
    summaries = mapping.get("summaries", {})

    def source_summary(kind: str, index: int) -> str | None:
        identifier = index if kind == "node" else original_ids[index]
        value = summaries.get(kind, {}).get(identifier)
        return None if value is None else str(value)

    graph = Graph(node_count=len(node_texts), sources=sources, targets=targets,
                  text=lambda kind, index: node_texts[index] if kind == "node" else edge_texts[index],
                  kind=lambda index: kinds[index], name=lambda index: names[index], domain=domain,
                  original_edge_ids=original_ids, node_kinds=node_kinds,
                  summary=source_summary if summaries else None)
    schema = {"domain": domain, "directed": True, "weighted": False,
              "node_count": graph.node_count, "edge_count": graph.edge_count,
              "node_kinds": list(node_kinds), "text_fields": ["text"],
              "parallel_edges": "distinct original records",
              "text_semantics": {"node": spec["node"], "edge": spec["edge"]},
              "edge_direction": "stored source to target; original ID restored only at output",
              "identity_lookup": ("Native node names/titles; Amazon reviewer aliases come only from source-bound native review headers."
                                  if identity_aliases else "Native node names/titles and declared typed IDs."),
              "text_projection": spec["projection"]}
    return graph, schema

class _SafeState:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _Data(_SafeState):
    pass


class _DataEdgeAttr(_SafeState):
    pass


class _DataTensorAttr(_SafeState):
    pass


class _GlobalStorage(_SafeState):
    pass


def _safe_load_from_bytes(blob: bytes) -> Any:
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}") from exc


ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("torch_geometric.data.data", "Data"): _Data,
    ("torch_geometric.data.data", "DataEdgeAttr"): _DataEdgeAttr,
    ("torch_geometric.data.data", "DataTensorAttr"): _DataTensorAttr,
    ("torch_geometric.data.storage", "GlobalStorage"): _GlobalStorage,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.storage", "_load_from_bytes"): _safe_load_from_bytes,
    ("collections", "OrderedDict"): collections.OrderedDict,
}

def _twitter_author_projection(
    mapping: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], list[str]]:
    """Return the complete author-only projection without inspecting a query."""

    edge_index = mapping["edge_index"]
    raw_sources = tuple(int(value) for value in edge_index[0].tolist())
    raw_targets = tuple(int(value) for value in edge_index[1].tolist())
    raw_node_texts = [str(value) for value in mapping["text_nodes"]]
    raw_edge_texts = [str(value) for value in mapping["text_edges"]]
    if len(raw_sources) != len(raw_targets) or len(raw_sources) != len(raw_edge_texts):
        raise ValueError("Twitter edge arrays disagree")

    projected_node_texts = list(raw_node_texts)
    tweet_texts: dict[int, str] = {}
    for target, edge_text in zip(raw_targets, raw_edge_texts, strict=True):
        previous = tweet_texts.setdefault(target, edge_text)
        if previous != edge_text:
            raise ValueError(f"Twitter tweet {target} has conflicting incident text")
    for tweet_id, text_value in tweet_texts.items():
        if not 0 <= tweet_id < len(projected_node_texts):
            raise ValueError(f"Twitter tweet ID {tweet_id} is outside the node range")
        projected_node_texts[tweet_id] = text_value

    sources: list[int] = []
    targets: list[int] = []
    original_edge_ids: list[int] = []
    for original_edge_id, (source, target) in enumerate(
        zip(raw_sources, raw_targets, strict=True)
    ):
        if not 0 <= source < len(raw_node_texts) or not 0 <= target < len(raw_node_texts):
            raise ValueError("Twitter endpoint lies outside the node range")
        if raw_node_texts[source] != "user":
            continue
        if raw_node_texts[target] != "tweet":
            raise ValueError("Twitter authorship projection must target a tweet node")
        sources.append(source)
        targets.append(target)
        original_edge_ids.append(original_edge_id)
    if not sources:
        raise ValueError("Twitter authorship projection is empty")
    return (
        tuple(sources),
        tuple(targets),
        tuple(original_edge_ids),
        projected_node_texts,
    )

TEXT_SEMANTICS: dict[str, Any] = {'amazon_movie': {'node': {'item': 'full_text_unless_generic_placeholder', 'user': 'generic_actor_identity_only'}, 'edge': 'full_review_text_when_nonempty', 'projection': None}, 'citation': {'node': {'paper': 'full_text_when_nonempty'}, 'edge': 'full_text_when_nonempty', 'projection': None}, 'goodreads_children': {'node': {'book': 'full_text_unless_generic_placeholder', 'reader': 'generic_actor_identity_only'}, 'edge': 'full_review_text_when_nonempty', 'projection': None}, 'goodreads_crime': {'node': {'book': 'full_text_unless_generic_placeholder', 'reader': 'generic_actor_identity_only'}, 'edge': 'full_review_text_when_nonempty', 'projection': None}, 'goodreads_history': {'node': {'book': 'full_text_unless_generic_placeholder', 'reader': 'generic_actor_identity_only'}, 'edge': 'full_review_text_when_nonempty', 'projection': None}, 'reddit': {'node': {'subreddit': 'community_name_only_not_description', 'user': 'identity_name_only'}, 'edge': 'full_post_text_when_nonempty', 'projection': 'interleaved endpoint records -> one text per referenced node'}, 'twitter': {'node': {'tweet': 'full_tweet_text_shared_with_author_edge', 'user': 'generic_kind_only_no_independent_semantics'}, 'edge': 'full_tweet_text', 'projection': 'identical incoming edge text -> target tweet text; user nodes retain only kind'}}
