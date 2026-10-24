#!/usr/bin/env python3
"""Guard the reviewed IO organization of the 183 selected task revisions.

Semantic duplication and genuinely distinct views are decided in the per-task
review. This script pins that review and rejects detailed IO records returning
to reader files; it does not infer program equivalence or replay Lean proofs.
"""

import argparse
import json
from pathlib import Path
import re

from check_main_entry_roles import (
    ROOT, LEDGER, declarations, reader_files, sha, source_roots, without_comments,
)

MANIFEST = ROOT / "bench/review-checkpoints/2026-10-22/io-consolidation/audit.json"
IO_FIELDS = {
    "inputs", "outputs", "scratch", "buffers", "read", "write", "inputMask",
    "outputMask", "scratchWrite", "scratchMask", "schedule", "encode", "observe",
    "kernel", "projection", "inp", "in1", "in2", "out", "mask", "B",
    "read1", "read2",
}


def detailed_reader_records(path):
    """Conservative shape check, backed by a pinned full-file role review."""
    source = without_comments(path.read_text()).splitlines()
    decls = declarations(path)
    found = []
    for i, decl in enumerate(decls):
        if decl["kind"] not in {"def", "abbrev"}:
            continue
        end = decls[i + 1]["line"] - 1 if i + 1 < len(decls) else len(source)
        body = "\n".join(source[decl["line"] - 1:end])
        fields = IO_FIELDS & set(re.findall(r"\b(\w+)\s*:=", body))
        program_record = len(fields) >= 4 and {"read", "write"} <= fields
        masked_record = {"kernel", "projection", "out", "write"} <= fields and bool(
            {"read", "read1", "read2"} & fields)
        if program_record or masked_record:
            found.append({**decl, "fields": sorted(fields)})
    return found


def check(data, manifest):
    errors = []
    tasks = {r["task"]: r for r in data["tasks"]}
    reviews = {r["task"]: r for r in manifest["tasks"]}
    if len(manifest["tasks"]) != 183 or len(reviews) != 183 or set(reviews) != set(tasks):
        errors.append("IO review must cover exactly all 183 selected tasks")
    checked_readers, checked_io_sources = set(), set()
    for name in sorted(tasks.keys() & reviews.keys()):
        review = reviews[name]
        selected = tasks[name]["kernel_revision"]["public_entries"]
        if selected != review["selected_public_entries"]:
            errors.append(f"{name}: selected revision changed; IO review required")
        if review["status"] not in {"fixed", "no_issue_identified"} or review["remaining_issues"]:
            errors.append(f"{name}: unresolved IO organization finding")
        if not review.get("assessment"):
            errors.append(f"{name}: missing IO sharing/distinct-view assessment")
        expected_readers = {r["path"]: r for r in review["readers"]}
        actual = reader_files(selected, source_roots(selected))
        if {str(p.relative_to(ROOT)) for p in actual} != set(expected_readers):
            errors.append(f"{name}: reader/re-export set changed")
        for path in actual:
            rel = str(path.relative_to(ROOT))
            checked_readers.add(rel)
            pin = expected_readers.get(rel)
            if not pin or sha(path) != pin["sha256"]:
                errors.append(f"{name}: {rel}: reader changed; IO review required")
            records = detailed_reader_records(path)
            if records:
                errors.append(f"{name}: detailed IO record remains in {rel}: {records}")
        if not review.get("io_sources"):
            errors.append(f"{name}: no canonical/shared IO source recorded")
        for source in review.get("io_sources", []):
            path = ROOT / source["path"]
            checked_io_sources.add(source["path"])
            if not path.is_file() or sha(path) != source["sha256"]:
                errors.append(f"{name}: {source['path']}: IO data changed; review required")
        evidence = review["review_evidence"]
        path = ROOT / evidence["path"]
        if not path.is_file() or sha(path) != evidence["sha256"]:
            errors.append(f"{name}: review evidence pin differs")
        if review["status"] == "fixed" and not review.get("preservation_and_build_passed"):
            errors.append(f"{name}: IO move/reuse lacks preservation/build/joint evidence")
    return {
        "tasks": len(reviews), "readers": len(checked_readers),
        "io_source_files": len(checked_io_sources), "errors": errors,
        "status": "passed" if not errors else "failed",
        "scope": "Reviewed canonical IO sharing and reader placement; no full-source replay or GPU experiment",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Override the current ledger's IO review")
    args = parser.parse_args()
    data = json.loads(LEDGER.read_text())
    manifest = args.manifest or ROOT / data.get("io_organization_review", {}).get(
        "audit", str(MANIFEST.relative_to(ROOT)))
    result = check(data, json.loads(manifest.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
