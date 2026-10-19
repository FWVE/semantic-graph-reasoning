"""Full source topology and on-demand text; no query-specific preprocessing."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_graph_reasoning.data.citation import (
    abstract_from_text,
    load_citation_data,
    normalise_text,
    source_manifest,
    title_from_text,
)
from semantic_graph_reasoning.data.graph import CompactIntegerGraph
from semantic_graph_reasoning.logging_utils import stable_hash

REDDIT_MATERIALIZATION_SCHEMA = {
    "node": ["name", "node_kind", "text"],
    "edge": [
        "source",
        "target",
        "text",
        "source_name",
        "target_name",
        "source_kind",
        "target_kind",
    ],
}


REDDIT_DOMAIN_ADAPTER = {
    "domain_id": "tegdb_reddit_user_subreddit_graph",
    "node_semantics": (
        "The directed graph is bipartite: source-side nodes are Reddit users and "
        "target-side nodes are subreddits. A user-to-subreddit edge means that user "
        "made the represented post in that subreddit."
    ),
    "edge_semantics": (
        "Each directed edge is one concrete Reddit post interaction from a user to a "
        "subreddit; semantic conditions about what a post says belong to that edge."
    ),
    "node_kind_field": "node_kind distinguishes user from subreddit.",
    "node_kind_materialization_field": "node_kind",
    "name_field": "name is the exact user or subreddit name exposed by the dataset.",
    "directed_edge_endpoint_kinds": {
        "source": "user",
        "target": "subreddit",
    },
}


GOODREADS_CHILDREN_MATERIALIZATION_SCHEMA = {
    "node": ["title", "node_kind", "text"],
    "edge": [
        "source",
        "target",
        "text",
        "target_name",
    ],
}


GOODREADS_CHILDREN_DOMAIN_ADAPTER = {
    "domain_id": "tegdb_goodreads_children_reader_book_review_graph",
    "node_semantics": (
        "The directed graph is bipartite: source-side nodes are Goodreads readers and "
        "target-side nodes are children's books. A reader-to-book edge means that reader "
        "wrote the represented review of that book."
    ),
    "edge_semantics": (
        "Each directed edge is one concrete review from a reader to a book. Conditions "
        "about what a review says belong to the edge; conditions about a book's title, "
        "description, subject, or other book content belong to the book node."
    ),
    "node_kind_field": "node_kind distinguishes reader from book.",
    "node_kind_materialization_field": "node_kind",
    "name_field": (
        "name is `Reader <ID>` for reader nodes and the dataset book title for book nodes."
    ),
    "directed_edge_endpoint_kinds": {
        "source": "reader",
        "target": "book",
    },
}


BIPARTITE_REVIEW_MATERIALIZATION_SCHEMA = {
    "node": ["name", "title", "node_kind", "text"],
    "edge": [
        "source",
        "target",
        "text",
        "source_name",
        "target_name",
        "source_kind",
        "target_kind",
    ],
}


TWITTER_MATERIALIZATION_SCHEMA = {
    "node": ["name", "node_kind", "text"],
    "edge": [
        "source",
        "target",
        "text",
        "source_name",
        "target_name",
        "source_kind",
        "target_kind",
    ],
}


TWITTER_DOMAIN_ADAPTER = {
    "domain_id": "tegdb_twitter_author_tweet_graph",
    "node_semantics": (
        "The directed graph is bipartite: source-side nodes are Twitter users and "
        "target-side nodes are tweets. Only author-user to tweet records are graph "
        "edges in this projection."
    ),
    "edge_semantics": (
        "Each directed edge is one author-to-tweet relation. Mentioned-user records "
        "from the source are not authorship and are excluded from graph topology. "
        "The tweet text is exposed on the tweet node and the author edge."
    ),
    "node_kind_field": "node_kind distinguishes user from tweet.",
    "node_kind_materialization_field": "node_kind",
    "name_field": (
        "name is `User <ID>` for users and `Tweet <ID>: <text prefix>` for tweets."
    ),
    "directed_edge_endpoint_kinds": {
        "source": "user",
        "target": "tweet",
    },
    "edge_id_semantics": (
        "Every graph and evidence edge ID is the same dense framework ID for "
        "the authorship projection."
    ),
}


def _review_domain_adapter(
    *,
    domain_id: str,
    left_kind: str,
    right_kind: str,
    right_description: str,
) -> dict[str, Any]:
    return {
        "domain_id": domain_id,
        "node_semantics": (
            f"The directed graph is bipartite: source-side nodes are {left_kind}s and "
            f"target-side nodes are {right_description}. A {left_kind}-to-{right_kind} "
            f"edge means that {left_kind} wrote the represented review of that {right_kind}."
        ),
        "edge_semantics": (
            f"Each directed edge is one concrete review from a {left_kind} to a "
            f"{right_kind}. Conditions about what a review says belong to the edge; "
            f"conditions about the {right_kind}'s title or content belong to the "
            f"{right_kind} node."
        ),
        "node_kind_field": f"node_kind distinguishes {left_kind} from {right_kind}.",
        "node_kind_materialization_field": "node_kind",
        "name_field": (
            f"name is `<{left_kind.title()}> <ID>` for {left_kind} nodes and the "
            f"dataset title for {right_kind} nodes."
        ),
        "directed_edge_endpoint_kinds": {
            "source": left_kind,
            "target": right_kind,
        },
    }


BIPARTITE_REVIEW_PROFILES: dict[str, dict[str, Any]] = {
    "amazon_movie": {
        "dataset": "TEG-DB Amazon-Movie",
        "graph_id": "tegdb-amazon-movie",
        "left_kind": "user",
        "right_kind": "item",
        "domain_adapter": _review_domain_adapter(
            domain_id="tegdb_amazon_movie_user_item_review_graph",
            left_kind="user",
            right_kind="item",
            right_description="movie products",
        ),
    },
    "goodreads_history": {
        "dataset": "TEG-DB Goodreads-History",
        "graph_id": "tegdb-goodreads-history",
        "left_kind": "reader",
        "right_kind": "book",
        "domain_adapter": _review_domain_adapter(
            domain_id="tegdb_goodreads_history_reader_book_review_graph",
            left_kind="reader",
            right_kind="book",
            right_description="history books",
        ),
    },
    "goodreads_crime": {
        "dataset": "TEG-DB Goodreads-Crime",
        "graph_id": "tegdb-goodreads-crime",
        "left_kind": "reader",
        "right_kind": "book",
        "domain_adapter": _review_domain_adapter(
            domain_id="tegdb_goodreads_crime_reader_book_review_graph",
            left_kind="reader",
            right_kind="book",
            right_description="crime books",
        ),
    },
}


@dataclass(frozen=True)
class FrameworkDataset:
    """One query-independent graph/domain bundle consumed by the canonical runner."""

    profile: str
    compact_graph: CompactIntegerGraph
    content_provider: Any
    source_manifest: dict[str, Any]
    semantic_materialization_schema: Mapping[str, list[str]] | None = None
    domain_adapter: Mapping[str, Any] | None = None
    external_edge_ids: tuple[int, ...] | None = None


class CitationContentProvider:
    """Materialize only semantic fields explicitly requested by Inspector."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def materialize(
        self,
        *,
        entity_type: str,
        entity_id: str,
        permitted_fields: Sequence[str],
    ) -> dict[str, Any]:
        identifier = int(entity_id)
        available: dict[str, Any]
        if entity_type == "node":
            text = normalise_text(self.data.text_nodes[identifier])
            available = {
                "title": title_from_text(text),
                "abstract": abstract_from_text(text),
                "text": text,
            }
        elif entity_type == "edge":
            source_id = int(self.data.edge_index[0, identifier])
            target_id = int(self.data.edge_index[1, identifier])
            source_text = normalise_text(self.data.text_nodes[source_id])
            target_text = normalise_text(self.data.text_nodes[target_id])
            available = {
                "source": source_id,
                "target": target_id,
                "text": normalise_text(self.data.text_edges[identifier]),
                "source_title": title_from_text(source_text),
                "target_title": title_from_text(target_text),
            }
        else:
            raise ValueError(f"TEGDB adapter cannot materialize {entity_type!r}")
        missing = [field for field in permitted_fields if field not in available]
        if missing:
            raise ValueError("requested fields unavailable for entity: " + ", ".join(missing))
        return {field: available[field] for field in permitted_fields}


class RedditContentProvider:
    """Materialize exact user/subreddit names and post-local edge evidence."""

    def __init__(self, data: Any) -> None:
        self.data = data
        self._edges: dict[int, tuple[int, int, str]] = {}
        for (source_id, target_id), posts in data.posts.items():
            for post in posts:
                edge_id = int(post.edge_id)
                if edge_id in self._edges:
                    raise ValueError(f"duplicate Reddit edge ID {edge_id}")
                self._edges[edge_id] = (int(source_id), int(target_id), str(post.text))

    def materialize(
        self,
        *,
        entity_type: str,
        entity_id: str,
        permitted_fields: Sequence[str],
    ) -> dict[str, Any]:
        identifier = int(entity_id)
        available: dict[str, Any]
        if entity_type == "node":
            if identifier in self.data.user_name:
                node_kind = "user"
                name = str(self.data.user_name[identifier])
            elif identifier in self.data.subreddit_name:
                node_kind = "subreddit"
                name = str(self.data.subreddit_name[identifier])
            else:
                raise ValueError(f"unknown Reddit node ID {identifier}")
            available = {
                "name": name,
                "node_kind": node_kind,
                "text": f"{node_kind} {name}",
            }
        elif entity_type == "edge":
            try:
                source_id, target_id, text = self._edges[identifier]
            except KeyError as exc:
                raise ValueError(f"unknown Reddit edge ID {identifier}") from exc
            available = {
                "source": source_id,
                "target": target_id,
                "text": normalise_text(text),
                "source_name": str(self.data.user_name[source_id]),
                "target_name": str(self.data.subreddit_name[target_id]),
                "source_kind": "user",
                "target_kind": "subreddit",
            }
        else:
            raise ValueError(f"Reddit adapter cannot materialize {entity_type!r}")
        missing = [field for field in permitted_fields if field not in available]
        if missing:
            raise ValueError("requested fields unavailable for entity: " + ", ".join(missing))
        return {field: available[field] for field in permitted_fields}


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


class TwitterContentProvider:
    """Materialize author/tweet evidence under dense framework edge identity."""

    def __init__(
        self,
        mapping: Mapping[str, Any],
        *,
        original_edge_ids: tuple[int, ...],
        projected_node_texts: Sequence[str],
    ) -> None:
        self.mapping = mapping
        self.original_edge_ids = original_edge_ids
        self.projected_node_texts = projected_node_texts
        self._raw_sources = tuple(
            int(value) for value in mapping["edge_index"][0].tolist()
        )
        self._raw_targets = tuple(
            int(value) for value in mapping["edge_index"][1].tolist()
        )

    def _node_kind(self, identifier: int) -> str:
        return "tweet" if str(self.mapping["text_nodes"][identifier]) == "tweet" else "user"

    def _node_name(self, identifier: int) -> str:
        if self._node_kind(identifier) == "user":
            return f"User {identifier}"
        compact = normalise_text(self.projected_node_texts[identifier])
        return f"Tweet {identifier}: {compact[:72]}" if compact else f"Tweet {identifier}"

    def materialize(
        self,
        *,
        entity_type: str,
        entity_id: str,
        permitted_fields: Sequence[str],
    ) -> dict[str, Any]:
        identifier = int(entity_id)
        available: dict[str, Any]
        if entity_type == "node":
            if not 0 <= identifier < len(self.projected_node_texts):
                raise ValueError(f"unknown Twitter node ID {identifier}")
            available = {
                "name": self._node_name(identifier),
                "node_kind": self._node_kind(identifier),
                "text": normalise_text(self.projected_node_texts[identifier]),
            }
        elif entity_type == "edge":
            if not 0 <= identifier < len(self.original_edge_ids):
                raise ValueError(f"unknown Twitter projection edge ID {identifier}")
            original_edge_id = self.original_edge_ids[identifier]
            source = self._raw_sources[original_edge_id]
            target = self._raw_targets[original_edge_id]
            available = {
                "source": source,
                "target": target,
                "text": normalise_text(self.mapping["text_edges"][original_edge_id]),
                "source_name": self._node_name(source),
                "target_name": self._node_name(target),
                "source_kind": "user",
                "target_kind": "tweet",
            }
        else:
            raise ValueError(f"Twitter adapter cannot materialize {entity_type!r}")
        missing = [field for field in permitted_fields if field not in available]
        if missing:
            raise ValueError("requested fields unavailable for entity: " + ", ".join(missing))
        return {field: available[field] for field in permitted_fields}


def _goodreads_is_reader(value: Any) -> bool:
    return value == -1 or str(value).strip() == "-1"


_BOOK_TITLE_PATTERN = re.compile(r"This book tittled (.*?) is a ", re.DOTALL)


def _goodreads_book_title(value: Any, node_id: int) -> str:
    text = " ".join(str(value).split())
    match = _BOOK_TITLE_PATTERN.search(text)
    if match:
        return " ".join(match.group(1).split()).strip()
    return f"Book node {node_id}"


_AMAZON_ITEM_TITLE = re.compile(r"The product titled ['\"](.*?)['\"]\.", re.DOTALL)


class BipartiteReviewContentProvider:
    """Materialize identities and review-local evidence for a bipartite review graph."""

    def __init__(
        self,
        mapping: Mapping[str, Any],
        *,
        left_kind: str,
        right_kind: str,
        profile: str,
    ) -> None:
        self.mapping = mapping
        self.left_kind = left_kind
        self.right_kind = right_kind
        self.profile = profile

    def _node_kind(self, identifier: int) -> str:
        return (
            self.left_kind
            if _goodreads_is_reader(self.mapping["text_node_labels"][identifier])
            else self.right_kind
        )

    def _node_name(self, identifier: int) -> str:
        if self._node_kind(identifier) == self.left_kind:
            return f"{self.left_kind.title()} {identifier}"
        if self.profile == "amazon_movie":
            text = normalise_text(self.mapping["text_nodes"][identifier])
            match = _AMAZON_ITEM_TITLE.search(text)
            return normalise_text(match.group(1)) if match else f"Item {identifier}"
        return _goodreads_book_title(self.mapping["text_nodes"][identifier], identifier)

    def materialize(
        self,
        *,
        entity_type: str,
        entity_id: str,
        permitted_fields: Sequence[str],
    ) -> dict[str, Any]:
        identifier = int(entity_id)
        available: dict[str, Any]
        if entity_type == "node":
            kind = self._node_kind(identifier)
            text = normalise_text(self.mapping["text_nodes"][identifier])
            available = {
                "name": self._node_name(identifier),
                "title": self._node_name(identifier) if kind == self.right_kind else "",
                "node_kind": kind,
                "text": text,
            }
        elif entity_type == "edge":
            source_id = int(self.mapping["edge_index"][0, identifier])
            target_id = int(self.mapping["edge_index"][1, identifier])
            available = {
                "source": source_id,
                "target": target_id,
                "text": normalise_text(self.mapping["text_edges"][identifier]),
                "source_name": self._node_name(source_id),
                "target_name": self._node_name(target_id),
                "source_kind": self._node_kind(source_id),
                "target_kind": self._node_kind(target_id),
            }
        else:
            raise ValueError(f"Goodreads adapter cannot materialize {entity_type!r}")
        missing = [field for field in permitted_fields if field not in available]
        if missing:
            raise ValueError("requested fields unavailable for entity: " + ", ".join(missing))
        return {field: available[field] for field in permitted_fields}


class GoodreadsChildrenContentProvider(BipartiteReviewContentProvider):
    """Native Goodreads Children text fields."""

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        super().__init__(
            mapping,
            left_kind="reader",
            right_kind="book",
            profile="goodreads_children",
        )


def compact_reddit_graph(data: Any, *, source_hash: str) -> CompactIntegerGraph:
    """Convert every post edge without inspecting a query or candidate set."""

    edge_sources: list[int | None] = [None] * int(data.edge_count)
    edge_targets: list[int | None] = [None] * int(data.edge_count)
    node_ids: set[int] = set()
    for (source_id, target_id), posts in data.posts.items():
        source = int(source_id)
        target = int(target_id)
        node_ids.update((source, target))
        for post in posts:
            edge_id = int(post.edge_id)
            if not 0 <= edge_id < len(edge_sources):
                raise ValueError(f"Reddit edge ID {edge_id} is outside the dense edge range")
            if edge_sources[edge_id] is not None:
                raise ValueError(f"duplicate Reddit edge ID {edge_id}")
            edge_sources[edge_id] = source
            edge_targets[edge_id] = target
    if not node_ids or any(value is None for value in edge_sources + edge_targets):
        raise ValueError("Reddit graph does not cover a complete dense edge range")
    node_count = max(node_ids) + 1
    sources = tuple(int(value) for value in edge_sources if value is not None)
    targets = tuple(int(value) for value in edge_targets if value is not None)
    return CompactIntegerGraph(
        graph_id="tegdb-reddit",
        snapshot_id=stable_hash(
            {
                "dataset_sha256": source_hash,
                "node_count": node_count,
                "edge_count": len(sources),
                "representation": "compact_integer_graph_v1",
            }
        ),
        directed=True,
        weighted=False,
        node_count=node_count,
        edge_sources=sources,
        edge_targets=targets,
    )


def compact_twitter_graph(
    mapping: Mapping[str, Any], *, source_hash: str
) -> tuple[CompactIntegerGraph, tuple[int, ...], list[str]]:
    """Convert the complete author-only Twitter projection to dense edge IDs."""

    sources, targets, original_edge_ids, projected_node_texts = (
        _twitter_author_projection(mapping)
    )
    compact = CompactIntegerGraph(
        graph_id="tegdb-twitter-author-tweet",
        snapshot_id=stable_hash(
            {
                "dataset_sha256": source_hash,
                "node_count": len(projected_node_texts),
                "edge_count": len(sources),
                "original_edge_ids": list(original_edge_ids),
                "projection": "author_user_to_tweet_only",
                "representation": "compact_integer_graph_v1",
            }
        ),
        directed=True,
        weighted=False,
        node_count=len(projected_node_texts),
        edge_sources=sources,
        edge_targets=targets,
    )
    return compact, original_edge_ids, projected_node_texts


def compact_bipartite_review_graph(
    mapping: Mapping[str, Any],
    *,
    source_hash: str,
    graph_id: str,
    left_kind: str,
    right_kind: str,
) -> CompactIntegerGraph:
    """Convert every left-to-right review edge without query filtering."""

    edge_index = mapping["edge_index"]
    text_nodes = mapping["text_nodes"]
    text_edges = mapping["text_edges"]
    labels = mapping["text_node_labels"]
    if tuple(edge_index.shape) != (2, len(text_edges)):
        raise ValueError("Goodreads edge_index and text_edges disagree")
    sources = tuple(int(value) for value in edge_index[0].tolist())
    targets = tuple(int(value) for value in edge_index[1].tolist())
    for source, target in zip(sources, targets, strict=True):
        if not _goodreads_is_reader(labels[source]) or _goodreads_is_reader(labels[target]):
            raise ValueError(
                f"review graph must contain only {left_kind}-to-{right_kind} edges"
            )
    return CompactIntegerGraph(
        graph_id=graph_id,
        snapshot_id=stable_hash(
            {
                "dataset_sha256": source_hash,
                "node_count": len(text_nodes),
                "edge_count": len(sources),
                "representation": "compact_integer_graph_v1",
            }
        ),
        directed=True,
        weighted=False,
        node_count=len(text_nodes),
        edge_sources=sources,
        edge_targets=targets,
    )


def compact_goodreads_children_graph(
    mapping: Mapping[str, Any], *, source_hash: str
) -> CompactIntegerGraph:
    """Build the Goodreads Children reader-book topology."""

    return compact_bipartite_review_graph(
        mapping,
        source_hash=source_hash,
        graph_id="tegdb-goodreads-children",
        left_kind="reader",
        right_kind="book",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare_framework_dataset(profile: str, source_path: Path) -> FrameworkDataset:
    """Load one complete dataset through its audited query-independent adapter."""

    pickle_path = source_path
    if profile == "citation":
        data, object_count, selected_index = load_citation_data(pickle_path)
        manifest = source_manifest(
            pickle_path=pickle_path,
            data=data,
            object_count=object_count,
            selected_index=selected_index,
        )
        compact = compact_full_graph(data, source_hash=manifest["sha256"])
        return FrameworkDataset(
            profile=profile,
            compact_graph=compact,
            content_provider=CitationContentProvider(data),
            source_manifest=manifest,
        )
    if profile == "reddit":
        from semantic_graph_reasoning.data import reddit

        data = reddit.load_graph(pickle_path)
        source_hash = _sha256_file(pickle_path)
        compact = compact_reddit_graph(data, source_hash=source_hash)
        manifest = {
            "dataset": "TEG-DB Reddit",
            "pickle_path": str(pickle_path),
            "sha256": source_hash,
            "size_bytes": pickle_path.stat().st_size,
            "security": {
                "outer_pickle_global_allowlist": [
                    f"{module}.{name}" for module, name in sorted(reddit.ALLOWED_GLOBALS)
                ],
                "embedded_tensor_weights_only": True,
                "unrestricted_pickle_load": False,
            },
            "normalized_graph": {
                "user_nodes": len(data.user_name),
                "subreddit_nodes": len(data.subreddit_name),
                "node_id_space": compact.node_count,
                "edge_interactions": data.edge_count,
                "unique_user_subreddit_pairs": len(data.posts),
            },
        }
        return FrameworkDataset(
            profile=profile,
            compact_graph=compact,
            content_provider=RedditContentProvider(data),
            source_manifest=manifest,
            semantic_materialization_schema=REDDIT_MATERIALIZATION_SCHEMA,
            domain_adapter=REDDIT_DOMAIN_ADAPTER,
        )
    if profile == "twitter":
        from semantic_graph_reasoning.data import serialization as restricted_pickle

        with pickle_path.open("rb") as stream:
            obj = restricted_pickle.RestrictedUnpickler(stream).load()
        mapping = obj._store._mapping
        source_hash = _sha256_file(pickle_path)
        compact, original_edge_ids, projected_node_texts = compact_twitter_graph(
            mapping, source_hash=source_hash
        )
        raw_edge_count = len(mapping["text_edges"])
        manifest = {
            "dataset": "TEG-DB Twitter",
            "pickle_path": str(pickle_path),
            "sha256": source_hash,
            "size_bytes": pickle_path.stat().st_size,
            "security": {
                "outer_pickle_global_allowlist": [
                    f"{module}.{name}"
                    for module, name in sorted(restricted_pickle.ALLOWED_GLOBALS)
                ],
                "embedded_tensor_weights_only": True,
                "unrestricted_pickle_load": False,
            },
            "normalized_graph": {
                "node_count": compact.node_count,
                "raw_edge_records": raw_edge_count,
                "author_to_tweet_edges": compact.edge_count,
                "mentioned_user_records_excluded": raw_edge_count - compact.edge_count,
                "edge_id_projection": "dense graph edge ID -> original source edge ID",
            },
        }
        return FrameworkDataset(
            profile=profile,
            compact_graph=compact,
            content_provider=TwitterContentProvider(
                mapping,
                original_edge_ids=original_edge_ids,
                projected_node_texts=projected_node_texts,
            ),
            source_manifest=manifest,
            semantic_materialization_schema=TWITTER_MATERIALIZATION_SCHEMA,
            domain_adapter=TWITTER_DOMAIN_ADAPTER,
            external_edge_ids=original_edge_ids,
        )
    if profile == "goodreads_children":
        from semantic_graph_reasoning.data import serialization as goodreads
        from semantic_graph_reasoning.data import serialization as restricted_pickle

        mapping = goodreads.load_mapping(pickle_path)
        source_hash = _sha256_file(pickle_path)
        compact = compact_goodreads_children_graph(mapping, source_hash=source_hash)
        labels = mapping["text_node_labels"]
        reader_count = sum(_goodreads_is_reader(value) for value in labels)
        manifest = {
            "dataset": "TEG-DB Goodreads Children",
            "pickle_path": str(pickle_path),
            "sha256": source_hash,
            "size_bytes": pickle_path.stat().st_size,
            "security": {
                "outer_pickle_global_allowlist": [
                    f"{module}.{name}"
                    for module, name in sorted(restricted_pickle.ALLOWED_GLOBALS)
                ],
                "embedded_tensor_weights_only": True,
                "unrestricted_pickle_load": False,
            },
            "normalized_graph": {
                "reader_nodes": reader_count,
                "book_nodes": len(labels) - reader_count,
                "node_count": compact.node_count,
                "review_edges": compact.edge_count,
            },
        }
        return FrameworkDataset(
            profile=profile,
            compact_graph=compact,
            content_provider=GoodreadsChildrenContentProvider(mapping),
            source_manifest=manifest,
            semantic_materialization_schema=GOODREADS_CHILDREN_MATERIALIZATION_SCHEMA,
            domain_adapter=GOODREADS_CHILDREN_DOMAIN_ADAPTER,
        )
    if profile in BIPARTITE_REVIEW_PROFILES:
        from semantic_graph_reasoning.data import serialization as restricted_pickle
        from semantic_graph_reasoning.data import serialization as review_graph

        spec = BIPARTITE_REVIEW_PROFILES[profile]
        mapping = review_graph.load_mapping(pickle_path)
        source_hash = _sha256_file(pickle_path)
        compact = compact_bipartite_review_graph(
            mapping,
            source_hash=source_hash,
            graph_id=str(spec["graph_id"]),
            left_kind=str(spec["left_kind"]),
            right_kind=str(spec["right_kind"]),
        )
        labels = mapping["text_node_labels"]
        left_count = sum(_goodreads_is_reader(value) for value in labels)
        left_kind = str(spec["left_kind"])
        right_kind = str(spec["right_kind"])
        manifest = {
            "dataset": str(spec["dataset"]),
            "pickle_path": str(pickle_path),
            "sha256": source_hash,
            "size_bytes": pickle_path.stat().st_size,
            "security": {
                "outer_pickle_global_allowlist": [
                    f"{module}.{name}"
                    for module, name in sorted(restricted_pickle.ALLOWED_GLOBALS)
                ],
                "embedded_tensor_weights_only": True,
                "unrestricted_pickle_load": False,
            },
            "normalized_graph": {
                f"{left_kind}_nodes": left_count,
                f"{right_kind}_nodes": len(labels) - left_count,
                "node_count": compact.node_count,
                "review_edges": compact.edge_count,
            },
        }
        return FrameworkDataset(
            profile=profile,
            compact_graph=compact,
            content_provider=BipartiteReviewContentProvider(
                mapping,
                left_kind=left_kind,
                right_kind=right_kind,
                profile=profile,
            ),
            source_manifest=manifest,
            semantic_materialization_schema=BIPARTITE_REVIEW_MATERIALIZATION_SCHEMA,
            domain_adapter=spec["domain_adapter"],
        )
    raise ValueError(f"unknown dataset profile {profile!r}")


def compact_full_graph(data: Any, *, source_hash: str) -> CompactIntegerGraph:
    """Convert the complete graph without inspecting the current query."""

    edge_sources = tuple(int(value) for value in data.edge_index[0].tolist())
    edge_targets = tuple(int(value) for value in data.edge_index[1].tolist())
    return CompactIntegerGraph(
        graph_id="tegdb-citation",
        snapshot_id=stable_hash(
            {
                "dataset_sha256": source_hash,
                "node_count": len(data.text_nodes),
                "edge_count": len(edge_sources),
                "representation": "compact_integer_graph_v1",
            }
        ),
        directed=True,
        weighted=False,
        node_count=len(data.text_nodes),
        edge_sources=edge_sources,
        edge_targets=edge_targets,
    )
