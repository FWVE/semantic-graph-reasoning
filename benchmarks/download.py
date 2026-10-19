"""List or download the seven SemGraph-Bench source graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DATASET = "https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/resolve/baa547adc859e97e0845f77e36b98f863d1d96b2"


def source_records(root: Path = ROOT) -> list[dict]:
    registry = json.loads((root / "benchmarks/sources.json").read_text("utf-8"))
    records = []
    for source in registry["graph_sources"]:
        target = (root / source["path"]).resolve()
        if not target.is_relative_to((root / "benchmarks/sources").resolve()):
            raise ValueError("source path escaped benchmarks/sources")
        upstream = Path(source["upstream_path"])
        if upstream.is_absolute() or ".." in upstream.parts:
            raise ValueError("unexpected upstream source path")
        records.append({"domain": source["graph_instance"], "path": source["path"],
                        "size_bytes": source["size_bytes"],
                        "url": f"{DATASET}/{source['upstream_path']}"})
    return records


def download(record: dict, root: Path = ROOT) -> str:
    target = root / record["path"]
    if target.exists():
        if target.stat().st_size != record["size_bytes"]:
            raise ValueError("existing source differs; preserve and inspect it")
        return "already_present"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{uuid4().hex}.part")
    try:
        with urlopen(record["url"], timeout=120) as response, temporary.open("xb") as output:  # noqa: S310
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size != record["size_bytes"]:
            raise ValueError(f"download size mismatch: {record['domain']}")
        if target.exists():
            raise FileExistsError(f"source appeared during download: {target}")
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", help="Comma-separated graph domains; omit for all seven")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true")
    action.add_argument("--verify", action="store_true", help="Check local file presence and size")
    args = parser.parse_args()
    records = source_records()
    selected = set(args.domains.split(",")) if args.domains else {r["domain"] for r in records}
    if not selected <= {r["domain"] for r in records}:
        parser.error("unknown graph domain")
    for record in records:
        if record["domain"] not in selected:
            continue
        target = ROOT / record["path"]
        state = "present" if target.is_file() else "missing"
        if args.download:
            state = download(record)
        elif args.verify:
            if not target.is_file() or target.stat().st_size != record["size_bytes"]:
                raise ValueError(f"source missing or size mismatch: {record['domain']}")
            state = "verified"
        print(json.dumps({**record, "state": state}))


if __name__ == "__main__":
    main()
