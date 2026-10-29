#!/usr/bin/env python3
"""Guard the explicit per-task review of main formulas, domains and scope.

This pins a human semantic review and its targeted proof evidence. It does not
infer a mathematical specification from syntax or rerun the corpus proofs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from check_main_entry_roles import (
    ROOT, LEDGER, declarations, reader_files, sha, source_roots,
)

DEFAULT_MANIFEST = ROOT / "bench/review-checkpoints/2026-10-22/main-formulas/audit.json"
SEMANTIC_FIELDS = (
    "mathematical_target", "required_conditions", "scope",
    "canonical_connection", "fp_program_pair", "numerical_dependencies",
)
REVIEW_STATUSES = {"fixed", "unchanged", "substantive_gap"}
VERIFICATION_STATUSES = {
    "targeted_passed", "unchanged_reused", "passed_with_recorded_limit",
}


def check(data, manifest):
    errors = []
    tasks = {row["task"]: row for row in data["tasks"]}
    rows = manifest.get("tasks", [])
    reviews = {row["task"]: row for row in rows}
    if len(rows) != 183 or len(reviews) != 183 or set(reviews) != set(tasks):
        errors.append("Formula review must cover exactly the 183 selected tasks")
    statuses = Counter()
    readers_checked = 0
    for name in sorted(tasks.keys() & reviews.keys()):
        row = reviews[name]
        statuses[row.get("status", "missing")] += 1
        if row.get("status") not in REVIEW_STATUSES:
            errors.append(f"{name}: missing final review disposition")
        for field in SEMANTIC_FIELDS:
            if not row.get(field):
                errors.append(f"{name}: missing semantic review field {field}")
        remaining = row.get("remaining_issues", [])
        if bool(remaining) != (row.get("status") == "substantive_gap"):
            errors.append(f"{name}: gap disposition and remaining issues disagree")
        if remaining and not row.get("gap_evidence"):
            errors.append(f"{name}: substantive gap lacks concrete evidence")
        selected = tasks[name]["kernel_revision"]["public_entries"]
        if selected != row.get("public_entries"):
            errors.append(f"{name}: selected revision changed; renewed review required")
        actual = reader_files(selected, source_roots(selected))
        expected = {r["path"]: r for r in row.get("readers", [])}
        if {str(p.relative_to(ROOT)) for p in actual} != set(expected):
            errors.append(f"{name}: reviewed reader set differs")
        for path in actual:
            rel = str(path.relative_to(ROOT))
            pin = expected.get(rel)
            readers_checked += 1
            if not pin:
                continue
            if sha(path) != pin.get("sha256"):
                errors.append(f"{name}: {rel}: source changed; renewed review required")
            specs = [d for d in declarations(path) if d["kind"] == "specification"]
            if specs != pin.get("specifications"):
                errors.append(f"{name}: {rel}: headline inventory differs")
        for rel, expected_sha in row.get("source_sha256", {}).items():
            path = ROOT / rel
            if not path.is_file() or sha(path) != expected_sha:
                errors.append(f"{name}: {rel}: reviewed source/connection changed")
        if not row.get("source_sha256"):
            errors.append(f"{name}: missing reviewed source/connection pins")
        evidence = row.get("review_evidence", {})
        path = ROOT / evidence.get("path", "__missing_formula_review__")
        if not path.is_file() or sha(path) != evidence.get("sha256"):
            errors.append(f"{name}: semantic review evidence pin differs")
        verification = row.get("verification", {})
        if verification.get("status") not in VERIFICATION_STATUSES:
            errors.append(f"{name}: no usable proof verification disposition")
        if verification.get("status") == "passed_with_recorded_limit" and not verification.get("limits"):
            errors.append(f"{name}: verification limit lacks an explanation")
        for pin in verification.get("evidence_files", []):
            path = ROOT / pin["path"]
            if not path.is_file() or sha(path) != pin.get("sha256"):
                errors.append(f"{name}: {pin['path']}: proof evidence pin differs")
        ledger_review = tasks[name]["kernel_revision"].get("main_formula_review")
        if ledger_review:
            if ledger_review.get("status") != row.get("status"):
                errors.append(f"{name}: ledger and detailed review dispositions differ")
            if ledger_review.get("review_evidence") != row.get("review_evidence"):
                errors.append(f"{name}: ledger and detailed review evidence differ")
            if ledger_review.get("main_declarations") != row.get("current_main_declarations"):
                errors.append(f"{name}: ledger and current reviewed main declarations differ")
    for rel, expected_sha in manifest.get("rules_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected_sha:
            errors.append(f"{rel}: review policy changed; refresh explicit review record")
    for section in ("normalization", "independent_review", "shared_syntax_review"):
        evidence = manifest.get(section)
        if not evidence:
            continue
        path = ROOT / evidence["path"]
        if not path.is_file() or sha(path) != evidence.get("sha256"):
            errors.append(f"{section}: final aggregate review evidence pin differs")
    if manifest.get("counts") != dict(statuses):
        errors.append("Formula-review counts differ from task dispositions")
    return {
        "tasks": len(reviews), "readers": readers_checked,
        "counts": dict(statuses), "errors": errors,
        "status": "passed" if not errors else "failed",
        "scope": "Pinned per-task formula/domain/scope review and targeted proof evidence; no automatic semantic inference, full-source replay or GPU run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Use a specified formula-review manifest")
    args = parser.parse_args()
    data = json.loads(LEDGER.read_text())
    manifest = args.manifest or ROOT / data.get("main_formula_review", {}).get(
        "audit", str(DEFAULT_MANIFEST.relative_to(ROOT)))
    result = check(data, json.loads(manifest.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
