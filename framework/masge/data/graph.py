"""Compact integer graph shared by source adapters."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactIntegerGraph:
    """Complete integer graph with implicit, zero-based edge IDs."""

    graph_id: str
    snapshot_id: str
    directed: bool
    weighted: bool
    node_count: int
    edge_sources: tuple[int, ...]
    edge_targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("graph_id must not be blank")
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be blank")
        if self.node_count < 0:
            raise ValueError("node_count must be nonnegative")
        if len(self.edge_sources) != len(self.edge_targets):
            raise ValueError("edge_sources and edge_targets must be aligned")
        for endpoint in (*self.edge_sources, *self.edge_targets):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                raise ValueError("compact graph endpoints must be integers")
            if endpoint < 0 or endpoint >= self.node_count:
                raise ValueError("compact graph endpoint is outside node ID range")

    @property
    def edge_count(self) -> int:
        return len(self.edge_sources)
