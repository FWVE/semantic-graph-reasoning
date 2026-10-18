"""Verify the frozen benchmark locally, without models or original graphs."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from download import sha256, source_records
from reference import ARCHIVE, OUTPUT, ROOT, reference, resource_from_archive


def verify(root: Path = ROOT) -> dict:
    output = root / OUTPUT.relative_to(ROOT)
    manifest = json.loads((output / "delivery_manifest.json").read_text("utf-8"))
    if manifest["query_count"] != 450:
        raise ValueError("unexpected query count")
    for name, expected in manifest["files"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"delivery checksum mismatch: {name}")
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
    fingerprints = manifest["resource_and_evidence_fingerprints"]
    if len(fingerprints) != 450 or {f["query_id"] for f in fingerprints} != expected_ids:
        raise ValueError("incomplete or duplicate evidence fingerprints")
    with zipfile.ZipFile(root / ARCHIVE.relative_to(ROOT)) as archive:
        if archive.testzip() is not None:
            raise ValueError("corrupt evidence archive")
        for name in ("queries.csv", "gold.json", "audit.json"):
            if archive.read(name) != (output / name).read_bytes():
                raise ValueError(f"archive and consumer files differ: {name}")
        if archive.read("benchmarks/resources/sources.json") != (root / "benchmarks/sources.json").read_bytes():
            raise ValueError("source registry differs from frozen evidence")
        for fingerprint in fingerprints:
            qid = fingerprint["query_id"]
            resource = resource_from_archive(archive, qid)
            # Canonical JSON encoding used by the frozen resource fingerprints.
            canonical = (json.dumps(resource, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            if hashlib.sha256(canonical).hexdigest() != fingerprint["resource_sha256"]:
                raise ValueError(f"resource fingerprint mismatch: {qid}")
            raw = archive.read(f"benchmarks/ready/{qid}.json")
            if hashlib.sha256(raw).hexdigest() != fingerprint["ready_sha256"]:
                raise ValueError(f"ready fingerprint mismatch: {qid}")
            candidates, answers, _ = reference(resource, gold[qid])
            totals.update(candidates=len(candidates), answers=len(answers), no_solution=int(not answers))
            families[resource["reader"]["topology"]["structural_family"]] += 1
    if totals != {"candidates": 6683, "answers": 1844, "no_solution": 10} or sorted(families.values()) != [50] * 9:
        raise ValueError("frozen benchmark totals differ")
    records = source_records(root)
    return {"status": "verified", "benchmark_name": "SemGraph-Bench", "queries": len(public),
            "families": dict(sorted(families.items())), **totals, "registered_sources": len(records),
            "external_calls": 0, "original_graphs_loaded": False,
            "scope": "Integrity and frozen structure/answer consistency; no new semantic admission."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
