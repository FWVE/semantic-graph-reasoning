"""Check benchmark contents locally, without models or original graphs."""
from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

from download import source_records
from reference import ARCHIVE, OUTPUT, ROOT, reference, resource_from_archive


def verify(root: Path = ROOT) -> dict:
    output = root / OUTPUT.relative_to(ROOT)
    with (output / "queries.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", "graph_domain", "query", "anchor_node_ids"]:
            raise ValueError("public input columns differ")
        public = list(reader)
    expected_ids = {f"Q{i:04d}" for i in range(1, 451)}
    gold_rows = json.loads((output / "gold.json").read_text("utf-8"))
    gold = {r["query_id"]: r["certificate"] for r in gold_rows}
    if (len(public) != 450 or {r["query_id"] for r in public} != expected_ids
            or len(gold_rows) != 450 or gold.keys() != expected_ids):
        raise ValueError("public and reference QIDs differ")
    totals = Counter()
    families = Counter()
    with zipfile.ZipFile(root / ARCHIVE.relative_to(ROOT)) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or archive.testzip() is not None:
            raise ValueError("duplicate or corrupt archive members")
        for folder in ("ready", "resources"):
            expected = {f"benchmarks/{folder}/{qid}.json" for qid in expected_ids}
            actual = {name for name in members if name.startswith(f"benchmarks/{folder}/Q")}
            if actual != expected:
                raise ValueError(f"incomplete or unexpected {folder} records")
        for name in ("queries.csv", "gold.json"):
            if archive.read(name) != (output / name).read_bytes():
                raise ValueError(f"archive and consumer files differ: {name}")
        if archive.read("benchmarks/resources/sources.json") != (root / "benchmarks/sources.json").read_bytes():
            raise ValueError("source registry differs from archived evidence")
        for qid in sorted(expected_ids):
            resource = resource_from_archive(archive, qid)
            ready = json.loads(archive.read(f"benchmarks/ready/{qid}.json"))
            if (ready["query"]["query_id"] != qid or ready["certificate"]["query_id"] != qid
                    or ready["certificate"] != gold[qid]):
                raise ValueError(f"annotation and reference answers differ: {qid}")
            candidates, answers, _ = reference(resource, gold[qid])
            totals.update(candidates=len(candidates), answers=len(answers), no_solution=int(not answers))
            families[resource["reader"]["topology"]["structural_family"]] += 1
    if totals != {"candidates": 6683, "answers": 1844, "no_solution": 10} or sorted(families.values()) != [50] * 9:
        raise ValueError("benchmark totals differ")
    records = source_records(root)
    return {"status": "verified", "benchmark_name": "SemGraph-Bench", "queries": len(public),
            "families": dict(sorted(families.items())), **totals, "registered_sources": len(records),
            "external_calls": 0, "original_graphs_loaded": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
