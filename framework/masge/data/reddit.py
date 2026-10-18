"""Reddit records with original user, community and edge identities."""
from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

from masge.data.serialization import ALLOWED_GLOBALS, RestrictedUnpickler

__all__ = ["ALLOWED_GLOBALS", "load_graph"]

class PreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Post:
    edge_id: int
    text: str


@dataclass
class RedditGraph:
    user_name: dict[int, str]
    subreddit_name: dict[int, str]
    posts: dict[tuple[int, int], list[Post]]
    user_to_subreddits: dict[int, set[int]]
    subreddit_to_users: dict[int, set[int]]
    edge_count: int
    text_node_count: int


def _subreddit(value: str) -> str:
    return value[len("subreddit ") :].strip() if value.startswith("subreddit ") else value.strip()


def _user(value: str) -> str:
    compact = value[5:] if value.startswith("user ") else value
    return compact.split(" has flair ", 1)[0].strip()


def load_graph(path: Path) -> RedditGraph:
    with path.open("rb") as stream:
        obj = RestrictedUnpickler(stream).load()
    mapping = obj._store._mapping
    edge_index = mapping["edge_index"]
    text_nodes = mapping["text_nodes"]
    text_edges = mapping["text_edges"]
    if tuple(edge_index.shape) != (2, len(text_edges)) or len(text_nodes) != 2 * len(text_edges):
        raise PreparationError("Reddit pickle does not match the audited layout")

    user_name: dict[int, str] = {}
    subreddit_name: dict[int, str] = {}
    posts: dict[tuple[int, int], list[Post]] = collections.defaultdict(list)
    user_to_subreddits: dict[int, set[int]] = collections.defaultdict(set)
    subreddit_to_users: dict[int, set[int]] = collections.defaultdict(set)
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True)):
        source = int(source)
        target = int(target)
        sub_name = _subreddit(str(text_nodes[2 * edge_id]))
        user_name_value = _user(str(text_nodes[2 * edge_id + 1]))
        if target in subreddit_name and subreddit_name[target] != sub_name:
            raise PreparationError(f"Conflicting subreddit names for {target}")
        if source in user_name and user_name[source] != user_name_value:
            raise PreparationError(f"Conflicting usernames for {source}")
        subreddit_name[target] = sub_name
        user_name[source] = user_name_value
        posts[(source, target)].append(Post(edge_id=edge_id, text=str(text_edges[edge_id])))
        user_to_subreddits[source].add(target)
        subreddit_to_users[target].add(source)
    return RedditGraph(
        user_name=user_name,
        subreddit_name=subreddit_name,
        posts=dict(posts),
        user_to_subreddits=dict(user_to_subreddits),
        subreddit_to_users=dict(subreddit_to_users),
        edge_count=len(text_edges),
        text_node_count=len(text_nodes),
    )
