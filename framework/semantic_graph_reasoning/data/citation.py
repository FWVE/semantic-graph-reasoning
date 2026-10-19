"""Citation source loading and native text fields."""
from __future__ import annotations

import hashlib
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO


class SourceDataError(RuntimeError):
    """A readable error caused by malformed source data."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalise_text(value: Any) -> str:
    """Represent a source text field faithfully enough for deterministic views."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


_TITLE_PATTERN = re.compile(r"^The paper titled ['\"](?P<title>.+?)['\"] presents\s*", re.DOTALL)


def title_from_text(text: str) -> str:
    match = _TITLE_PATTERN.match(text)
    if match:
        return " ".join(match.group("title").split())
    first_line = " ".join(text.split())
    return first_line[:180] if first_line else "Untitled paper"


def abstract_from_text(text: str) -> str:
    match = _TITLE_PATTERN.match(text)
    if match:
        return text[match.end() :].strip()
    return text.strip()


def _data_attribute(data: Any, name: str) -> Any:
    if not hasattr(data, name):
        raise SourceDataError(f"Loaded object lacks required attribute {name!r}")
    return getattr(data, name)


def load_citation_data_stream(stream: BinaryIO) -> tuple[Any, int, int]:
    """Deserialize one trusted PyG object stream and verify field alignment.

    The caller is responsible for explicitly opting in before calling this
    function because Python pickle deserialization can execute code.
    """

    try:
        loaded = pickle.load(stream)  # noqa: S301 - trusted registered source.
    except ModuleNotFoundError as error:
        raise SourceDataError(
            "The selected Conda environment needs torch and torch_geometric to load this "
            "trusted PyG pickle."
        ) from error

    if isinstance(loaded, (list, tuple)):
        if len(loaded) != 1:
            raise SourceDataError(
                "Expected one TEG-DB Data object or a one-item sequence; "
                f"received a sequence of length {len(loaded)}."
            )
        data = loaded[0]
        object_count = len(loaded)
        selected_index = 0
    else:
        data = loaded
        object_count = 1
        selected_index = 0

    edge_index = _data_attribute(data, "edge_index")
    text_nodes = _data_attribute(data, "text_nodes")
    text_edges = _data_attribute(data, "text_edges")
    node_labels = _data_attribute(data, "node_labels")
    shape = getattr(edge_index, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != 2:
        raise SourceDataError("edge_index must have shape [2, number_of_edges]")
    edge_count = int(shape[1])
    if not isinstance(text_nodes, Sequence) or isinstance(text_nodes, (str, bytes)):
        raise SourceDataError("text_nodes must be a sequence")
    if not isinstance(text_edges, Sequence) or isinstance(text_edges, (str, bytes)):
        raise SourceDataError("text_edges must be a sequence")
    if not isinstance(node_labels, Sequence) or isinstance(node_labels, (str, bytes)):
        raise SourceDataError("node_labels must be a sequence")
    if len(text_nodes) != len(node_labels):
        raise SourceDataError("text_nodes and node_labels have different lengths")
    if len(text_edges) != edge_count:
        raise SourceDataError(
            "text_edges must align one-for-one with edge_index columns: "
            f"{len(text_edges)} != {edge_count}"
        )
    return data, object_count, selected_index


def load_citation_data(pickle_path: Path) -> tuple[Any, int, int]:
    """Open one trusted Citation pickle and delegate to the stream loader."""

    with pickle_path.open("rb") as stream:
        return load_citation_data_stream(stream)


def source_manifest(
    *, pickle_path: Path, data: Any, object_count: int, selected_index: int
) -> dict[str, Any]:
    edge_index = _data_attribute(data, "edge_index")
    nodes = _data_attribute(data, "text_nodes")
    return {
        "dataset": "TEG-DB Citation",
        "dataset_path": str(pickle_path.resolve()),
        "file_type": "pickle (PyG Data object)",
        "object_count": object_count,
        "selected_data_object_index": selected_index,
        "number_of_nodes": len(nodes),
        "number_of_edges": int(edge_index.shape[1]),
        "sha256": sha256_file(pickle_path),
        "node_labels_role": "metadata_only_not_used_for_semantic_conditions",
    }
