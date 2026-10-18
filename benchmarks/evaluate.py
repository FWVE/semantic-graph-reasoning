"""Offline SemGraph-Bench scoring after freezing predictions; no model or graph calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

from download import sha256
from reference import ARCHIVE, OUTPUT, key, reference, resource_from_archive
from verify import verify

PROTOCOL = "semgraph-complete-set-v1-ordered-paths"


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def native_path(group: dict, plan: dict) -> list[int] | None:
    """Recover an explicit path from native output and its own Plan, without references."""
    if plan.get("operation") == "shortest_paths":
        return group["node_ids"]
    if plan.get("operation") != "pattern" or plan.get("direction") == "undirected":
        return None
    paths = set()
    for assignment in group.get("assignments", []):
        roles = assignment["role_node_ids"]
        links = [(roles[r["source"]], roles[r["target"]]) for r in plan["relations"]]
        successors = {source: target for source, target in links}
        if len(successors) != len(links) or len(set(successors.values())) != len(links):
            return None
        if set(successors.values()) - set(successors) == set() or len(links) != len(group["node_ids"]) - 1:
            return None
        visited = [group["anchor_node_id"]]
        while visited[-1] in successors:
            node = successors[visited[-1]]
            if node in visited:
                return None
            visited.append(node)
        if set(visited) != set(group["node_ids"]):
            return None
        emitted_edges = {e for values in assignment["relation_edge_ids"].values() for e in values}
        if emitted_edges != set(group["edge_ids"]):
            return None
        paths.add(tuple(visited))
    return list(next(iter(paths))) if len(paths) == 1 else None


def normalize_native(result: dict, workspace: dict) -> dict:
    result = deepcopy(result)
    if result.get("status") == "complete" and isinstance(result.get("answers"), list):
        plan = workspace.get("plan", {})
        for group in result["answers"]:
            try:
                path = native_path(group, plan)
            except (KeyError, TypeError, ValueError):
                path = None
            if path is not None:
                group["path_node_ids"] = path
    return result


def predicted_key(group: object, is_path: bool) -> tuple:
    if not isinstance(group, dict) or type(group.get("anchor_node_id")) is not int:
        raise ValueError("answer requires an integer anchor_node_id")
    for field in ("node_ids", "edge_ids"):
        values = group.get(field)
        if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"{field} must be a list of nonnegative original IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate IDs inside {field}")
    if not group["node_ids"] or group["anchor_node_id"] not in group["node_ids"]:
        raise ValueError("answer must contain its anchor")
    path = group.get("path_node_ids") if is_path else None
    if is_path and (not isinstance(path, list) or any(type(n) is not int for n in path)
                    or len(path) != len(group["node_ids"]) or set(path) != set(group["node_ids"])
                    or path[0] != group["anchor_node_id"]):
        raise ValueError("path answer requires the full ordered path_node_ids from its anchor")
    return key(group["anchor_node_id"], group["node_ids"], group["edge_ids"], path)


def score(result: object, structural: set, expected: set, is_path: bool) -> dict:
    complete = isinstance(result, dict) and result.get("status") == "complete" and isinstance(result.get("answers"), list)
    groups = result["answers"] if complete else []
    parsed, invalid = {}, []
    duplicates = 0
    for index, group in enumerate(groups):
        try:
            identity = predicted_key(group, is_path)
        except (ValueError, TypeError, KeyError) as error:
            invalid.append({"index": index, "reason": str(error)})
            continue
        duplicates += int(identity in parsed)
        parsed.setdefault(identity, group)
    identities = set(parsed)
    correct = identities & expected
    structurally_correct = identities & structural
    valid = complete and not invalid
    precision = len(correct) / len(groups) if groups else float(valid and not expected)
    recall = len(correct) / len(expected) if expected else float(valid and not groups)
    f1 = 2 * len(correct) / (len(groups) + len(expected)) if groups or expected else float(valid)
    return {"status": result.get("status", "invalid") if isinstance(result, dict) else "invalid",
            "valid_output": bool(valid), "predicted_count": len(groups), "gold_count": len(expected),
            "duplicate_count": duplicates, "invalid_answers": invalid,
            "answer_set_exact_match": bool(valid and not duplicates and identities == expected),
            "precision": precision, "recall": recall, "f1": f1,
            "structural_correct_count": len(structurally_correct), "correct_count": len(correct),
            "structural_correct_subgraphs": [parsed[k] for k in sorted(structurally_correct)],
            "correct_subgraphs": [parsed[k] for k in sorted(correct)]}


def read_frozen(path: Path, fingerprints: dict) -> object:
    raw = path.read_bytes()
    fingerprints[str(path.resolve())] = hashlib.sha256(raw).hexdigest()

    def unique_object(pairs: list) -> dict:
        value = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"duplicate JSON field: {name}")
            value[name] = item
        return value

    return json.loads(raw, object_pairs_hook=unique_object)


def campaign_predictions(campaign: Path, phase: str, fingerprints: dict) -> dict:
    ledger = read_frozen(campaign / "ledger.json", fingerprints)
    selected = {}
    for run in sorted(ledger["runs"], key=lambda row: row["run"]):
        if run["phase"] == phase:
            selected[run["query_id"]] = run
    predictions = {}
    for qid, run in selected.items():
        directory = (campaign / run["path"]).resolve()
        if not directory.is_relative_to(campaign.resolve()):
            raise ValueError("ledger run path escaped campaign")
        path = directory / "result.json"
        if run["status"] == "started" or not path.is_file():
            raise ValueError(f"unsettled latest attempt: {qid}; settle it before scoring")
        result = read_frozen(path, fingerprints)
        workspace_path = directory / "workspace.json"
        workspace = read_frozen(workspace_path, fingerprints) if workspace_path.is_file() else {}
        predictions[qid] = normalize_native(result, workspace)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="JSON object mapping QID to status/answers")
    source.add_argument("--campaign", type=Path, help="Native campaign; latest attempt per QID in selected phase")
    parser.add_argument("--phase", choices=["debug", "acceptance"], default="acceptance")
    parser.add_argument("--query-ids", help="Explicit subset; default denominator is all 450 questions")
    parser.add_argument("--output", type=Path, required=True, help="New report directory; existing directories are refused")
    args = parser.parse_args()
    fingerprints = {}
    predictions = (read_frozen(args.predictions, fingerprints) if args.predictions else
                   campaign_predictions(args.campaign, args.phase, fingerprints))
    if not isinstance(predictions, dict):
        parser.error("predictions must map each query ID to its result")
    with (OUTPUT / "queries.csv").open(encoding="utf-8-sig", newline="") as stream:
        public_ids = {r["query_id"] for r in csv.DictReader(stream)}
    selected = args.query_ids.split(",") if args.query_ids else sorted(public_ids)
    if (not predictions.keys() <= public_ids or not set(selected) <= public_ids
            or len(selected) != len(set(selected))):
        parser.error("unknown or duplicate query IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "prediction_freeze.json", {"protocol": PROTOCOL, "query_ids": selected,
          "files": fingerprints, "evaluator_sha256": sha256(Path(__file__)),
          "reference_helper_sha256": sha256(Path(__file__).with_name("reference.py")),
          "selection": "latest attempt per QID within phase" if args.campaign else "explicit predictions",
          "phase": args.phase if args.campaign else None})
    write(args.output / "predictions.frozen.json", predictions)
    # The complete prediction snapshot is now on disk. First private reference access follows.
    verify()
    with zipfile.ZipFile(ARCHIVE) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    for name in ("gold.json", "queries.csv"):
        if sha256(OUTPUT / name) != manifest["files"][name]:
            raise ValueError(f"reference input checksum mismatch: {name}")
    gold = {r["query_id"]: r["certificate"] for r in json.loads((OUTPUT / "gold.json").read_text("utf-8"))}
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for qid in selected:
            structural, expected, is_path = reference(resource_from_archive(archive, qid), gold[qid])
            result = predictions.get(qid, {"status": "missing", "answers": None})
            rows.append({"query_id": qid, **score(result, structural, expected, is_path)})
    summary = {"protocol": PROTOCOL, "queries": len(rows), "submitted": len(set(selected) & predictions.keys()),
               "exact_matches": sum(r["answer_set_exact_match"] for r in rows),
               "accuracy": sum(r["answer_set_exact_match"] for r in rows) / len(rows),
               "macro_f1": sum(r["f1"] for r in rows) / len(rows),
               "gold_sha256": sha256(OUTPUT / "gold.json"), "evidence_zip_sha256": sha256(ARCHIVE),
               "external_calls": 0, "benchmark_name": "SemGraph-Bench"}
    write(args.output / "per_query.json", rows)
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
