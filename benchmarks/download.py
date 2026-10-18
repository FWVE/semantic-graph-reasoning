"""List or fetch the seven pinned SemGraph-Bench source graphs without unpickling."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def source_records(root: Path = ROOT) -> list[dict]:
    """Cross-check the solver's source registry against pinned download provenance."""
    registry = json.loads((root / "benchmarks/sources.json").read_text("utf-8"))
    records = []
    for spec in registry["graph_sources"]:
        target = (root / spec["path"]).resolve()
        if not target.is_relative_to((root / "benchmarks/sources").resolve()):
            raise ValueError("source path escaped benchmarks/sources")
        manifest_path = root / spec["manifest_path"]
        dataset_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text("utf-8"))
        candidates = [r for r in manifest["files"] if (dataset_root / r["local_path"]).resolve() == target]
        if (len(candidates) != 1 or candidates[0]["sha256"] != spec["sha256"]
                or manifest["revision"] != spec["revision"]):
            raise ValueError(f"download provenance disagrees with registry: {spec['graph_instance']}")
        record = candidates[0]
        if (manifest["repo_id"] != "ZhuofengLi/TEG-Datasets"
                or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
                or ".." in Path(record["upstream_path"]).parts):
            raise ValueError("unexpected source repository, revision, or upstream path")
        records.append({"domain": spec["graph_instance"], "path": spec["path"],
                        "sha256": spec["sha256"], "size_bytes": record["size_bytes"],
                        "url": f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/"
                               f"{manifest['revision']}/{record['upstream_path']}"})
    return records


def download(record: dict, root: Path = ROOT) -> str:
    target = root / record["path"]
    if target.exists():
        if target.stat().st_size != record["size_bytes"] or sha256(target) != record["sha256"]:
            raise ValueError(f"existing source differs; preserve and inspect it: {target}")
        return "already_verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{uuid4().hex}.part")
    try:
        with urlopen(record["url"], timeout=120) as response, temporary.open("xb") as output:  # noqa: S310
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size != record["size_bytes"] or sha256(temporary) != record["sha256"]:
            raise ValueError(f"download checksum mismatch: {record['domain']}")
        # Refuse to replace a source another process installed while downloading.
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "downloaded_and_verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", help="Comma-separated domains; omitted means all seven")
    parser.add_argument("--download", action="store_true", help="Fetch missing files; default only lists them")
    parser.add_argument("--verify", action="store_true", help="Hash existing raw files; no deserialization")
    args = parser.parse_args()
    records = source_records()
    selected = set(args.domains.split(",")) if args.domains else {r["domain"] for r in records}
    if not selected <= {r["domain"] for r in records}:
        parser.error("unknown graph domain")
    for record in records:
        if record["domain"] not in selected:
            continue
        state = "present" if (ROOT / record["path"]).is_file() else "missing"
        if args.download:
            state = download(record)
        elif args.verify:
            target = ROOT / record["path"]
            if not target.is_file() or target.stat().st_size != record["size_bytes"] or sha256(target) != record["sha256"]:
                raise ValueError(f"source missing or checksum mismatch: {record['domain']}")
            state = "verified"
        print(json.dumps({**record, "state": state}))


if __name__ == "__main__":
    main()
