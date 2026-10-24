#!/usr/bin/env python3
"""Check selected reader organization against an explicit human role review.

This is a drift/import guard, not a proof or an automatic semantic classifier.
Genuine multi-kernel headlines are allowed and individually reviewed. Lean
build evidence and optional historical Misc specifications are separate.
"""

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "bench/tritonbench_g/task_progress.json"
MANIFEST = ROOT / "bench/review-checkpoints/2026-10-22/headline-separation/audit.json"
READER_STEMS = {"Correct", "FPEquiv", "FPEquivBound"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_comments(source):
    """Mask nested Lean comments/strings while preserving offsets and lines."""
    out = list(source)
    i, depth, quoted = 0, 0, False
    while i < len(source):
        if depth:
            if source.startswith("/-", i):
                out[i:i + 2] = "  "
                depth += 1
                i += 2
            elif source.startswith("-/", i):
                out[i:i + 2] = "  "
                depth -= 1
                i += 2
            else:
                if source[i] != "\n":
                    out[i] = " "
                i += 1
        elif quoted:
            if source[i] == "\\" and i + 1 < len(source):
                out[i:i + 2] = "  "
                i += 2
            else:
                quoted = source[i] != '"'
                if source[i] != "\n":
                    out[i] = " "
                i += 1
        elif source.startswith("/-", i):
            out[i:i + 2] = "  "
            depth = 1
            i += 2
        elif source.startswith("--", i):
            end = source.find("\n", i)
            end = len(source) if end < 0 else end
            out[i:end] = " " * (end - i)
            i = end
        elif source[i] == '"':
            quoted = True
            out[i] = " "
            i += 1
        else:
            i += 1
    return "".join(out)


@cache
def imports(path):
    return [module for match in re.finditer(
        r"^\s*(?:public\s+)?import\s+([^\n]+)", without_comments(path.read_text()), re.M
    ) for module in match.group(1).split()]


@cache
def declarations(path):
    source = without_comments(path.read_text())
    pattern = re.compile(
        r"^\s*(?:(?:private|protected|noncomputable|unsafe)\s+)*"
        r"(specification|theorem|lemma|alias|abbrev|def)\s+([^\s(:{\[]+)", re.M
    )
    return [{"kind": m.group(1), "name": m.group(2),
             "line": source.count("\n", 0, m.start(1)) + 1}
            for m in pattern.finditer(source)]


def source_roots(selected):
    # Frozen selected source overlays have precedence over the active tree.
    roots = []
    for name in selected:
        parts = Path(name).parts
        if parts[0] == ".veritile-cache" and "bench" in parts:
            root = ROOT.joinpath(*parts[:parts.index("bench")])
            if root not in roots:
                roots.append(root)
    return (*roots, ROOT)


@cache
def resolve(module, roots):
    rel = Path(*module.split(".")).with_suffix(".lean")
    return next((root / rel for root in roots if (root / rel).is_file()), None)


def reader_files(selected, roots):
    pending = [ROOT / p for p in selected if Path(p).stem in READER_STEMS]
    readers = set()
    while pending:
        path = pending.pop()
        if path in readers:
            continue
        readers.add(path)
        # Follow a re-export such as FPEquivBound -> FPEquiv explicitly.
        if not any(d["kind"] == "specification" for d in declarations(path)):
            for module in imports(path):
                target = resolve(module, roots)
                if target and target.stem in READER_STEMS:
                    pending.append(target)
    return sorted(readers)


def closure(paths, roots):
    pending, seen, reverse_edges = list(paths), set(), []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in imports(path):
            target = resolve(module, roots)
            if target is None:  # External Lean/Mathlib source is outside this guard.
                continue
            if "Proofs" in path.parts and target.stem in READER_STEMS:
                reverse_edges.append([str(path.relative_to(ROOT)), str(target.relative_to(ROOT))])
            pending.append(target)
    return seen, reverse_edges


def check(data, manifest):
    errors, headlines, readers_checked = [], 0, 0
    rows = {row["task"]: row for row in data["tasks"]}
    reviews = {row["task"]: row for row in manifest["tasks"]}
    if len(manifest["tasks"]) != 183 or len(reviews) != 183 or set(reviews) != set(rows):
        errors.append("Review must cover exactly the 183 selected tasks")
    for name in sorted(rows.keys() & reviews.keys()):
        review = reviews[name]
        selected = rows[name]["kernel_revision"]["public_entries"]
        if selected != review["selected_public_entries"]:
            errors.append(f"{name}: selected entries changed; role review required")
        roots = source_roots(selected)
        readers = reader_files(selected, roots)
        expected = {r["path"]: r for r in review["readers"]}
        actual = {str(p.relative_to(ROOT)) for p in readers}
        if actual != set(expected):
            errors.append(f"{name}: reader/re-export set differs from review")
        for path in readers:
            rel = str(path.relative_to(ROOT))
            readers_checked += 1
            if rel not in expected:
                continue
            approved = expected[rel]
            if sha(path) != approved["sha256"]:
                errors.append(f"{name}: {rel}: source changed; role review required")
            decls = declarations(path)
            if decls != [{k: d[k] for k in ("kind", "name", "line")}
                         for d in approved["declarations"]]:
                errors.append(f"{name}: {rel}: declaration inventory changed")
            for decl in approved["declarations"]:
                if not decl.get("role") or not decl.get("reason"):
                    errors.append(f"{name}: {rel}: declaration lacks role decision")
                if decl["kind"] in {"theorem", "lemma", "alias"}:
                    errors.append(f"{name}: {rel}: auxiliary proof remains in reader")
                if decl["kind"] == "specification":
                    headlines += 1
                    if decl["role"] not in {"independent_real", "actual_program_pair"}:
                        errors.append(f"{name}: {rel}: auxiliary marked as main specification")
            full_names = [d.get("full_name") for d in approved["declarations"]]
            if None in full_names or len(full_names) != len(set(full_names)):
                errors.append(f"{name}: {rel}: missing or duplicate qualified declaration metadata")
        for surface in ("Correct", "FPEquiv"):
            if not any(p.stem.startswith(surface) and any(
                    d["kind"] == "specification" for d in declarations(p)) for p in readers):
                errors.append(f"{name}: missing {surface} main specification")
        dependencies, reverse = closure(readers, roots)
        misc = sorted(str(p.relative_to(ROOT)) for p in dependencies if "Misc" in p.parts)
        if misc:
            errors.append(f"{name}: main imports Misc: {misc}")
        if reverse:
            errors.append(f"{name}: Proofs imports reader: {reverse}")
    return {"tasks": len(reviews), "readers": readers_checked, "headlines": headlines,
            "errors": errors, "status": "passed" if not errors else "failed",
            "scope": "Pinned semantic role review plus current repository import guard; no Lean replay or GPU run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Override the current ledger's role review")
    args = parser.parse_args()
    data = json.loads(LEDGER.read_text())
    manifest = args.manifest or ROOT / data.get("main_entry_role_review", {}).get(
        "audit", str(MANIFEST.relative_to(ROOT)))
    result = check(data, json.loads(manifest.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
