"""Restricted tensor deserialization for the registered source graphs."""
from __future__ import annotations

import collections
import io
import pickle
from pathlib import Path
from typing import Any, cast

import torch


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


ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("torch_geometric.data.data", "Data"): _Data,
    ("torch_geometric.data.data", "DataEdgeAttr"): _DataEdgeAttr,
    ("torch_geometric.data.data", "DataTensorAttr"): _DataTensorAttr,
    ("torch_geometric.data.storage", "GlobalStorage"): _GlobalStorage,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.storage", "_load_from_bytes"): _safe_load_from_bytes,
    ("collections", "OrderedDict"): collections.OrderedDict,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}") from exc


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        obj = RestrictedUnpickler(stream).load()
    mapping = obj._store._mapping
    required = {"edge_index", "text_nodes", "text_edges", "text_node_labels"}
    missing = required - set(mapping)
    if missing:
        raise RuntimeError(f"Missing required review-graph fields: {sorted(missing)}")
    return cast(dict[str, Any], mapping)
