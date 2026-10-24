#!/usr/bin/env python3
"""Inventory body-copy readiness for every selected task and worked example.

This is a source gate, not a Lean proof or a compiler-correctness certificate.
`--check` checks the recorded inventory, including pending entries, for drift.
`--require-complete` additionally rejects ANY pending entry. A partial manifest
cannot certify a multi-stage Python file. No variable renaming, cast deletion,
or generated Python mirror certifies a selected benchmark's original source.
New native counterparts of Lean-authored examples have a separate provenance.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from bench.audit_source import lean_code_mask
from bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("bench/source-correspondence.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"source outside repository: {name}")
    return path


def jit_functions(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text())
    result = []
    # Nested JITs are inventoried too; the pair checker currently supports only
    # explicitly named top-level functions and will refuse a nested selection.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
        if not any(ast.unparse(d) == "triton.jit" for d in decorators):
            continue
        casts = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if ((isinstance(call.func, ast.Attribute) and call.func.attr == "to"
                 and call.args and ast.unparse(call.args[0]) == "tl.int64")
                or (ast.unparse(call.func) == "tl.cast" and len(call.args) > 1
                    and ast.unparse(call.args[1]) == "tl.int64")):
                casts.append({"line": call.lineno, "expression": ast.unparse(call)})
        result.append({"function": node.name, "line": node.lineno,
                       "statements": len(node.body), "explicit_int64_casts": casts})
    return sorted(result, key=lambda f: f["line"])


def lean_features(path: Path) -> dict:
    code = lean_code_mask(path.read_text())
    patterns = {
        "quotations": r"\btriton\s*(?:\([^\n]*\))?\s*\{",
        "in_body_type_directives": r"\btl\.(?:region|param)\b",
        "antiquotes": r"\$\(",
        "dsl_assignment": r":=",
        "dsl_loop": r"\btl\.for\b",
        "dsl_numeric_helpers": r"\btl\.(?:toReal|welford)\b|\.round_to\s*\(",
        "load_dtype_extension": r"\bdtype\s*=",
    }
    # These are file-level screening counts (e.g. := also occurs in Lean
    # definitions, dtype= also in tl.full). They are NOT semantic defects.
    return {name: len(re.findall(pattern, code)) for name, pattern in patterns.items()}


def inspect(root: Path, task: str, python: str | None, lean: str, kind: str,
            provenance: str = "selected_revision") -> dict:
    row = {"name": task, "kind": kind, "python": python, "lean": lean}
    row["source_provenance"] = provenance
    hashes = {}
    lean_path = local(root, lean)
    if not lean_path.is_file():
        raise ValueError(f"missing selected Lean source: {lean}")
    hashes[lean] = digest(lean_path)
    row["lean_screen"] = lean_features(lean_path)
    code = lean_code_mask(lean_path.read_text())
    expected = []
    if python is not None:
        path = local(root, python)
        if not path.is_file():
            raise ValueError(f"missing selected Python source: {python}")
        hashes[python] = digest(path)
        row["python_jits"] = jit_functions(path)
        names = [fn["function"] for fn in row["python_jits"]]
        if len(names) != len(set(names)):
            raise ValueError(f"ambiguous selected Python JIT names: {python}")
        expected = [(python, name) for name in names]
    else:
        row["python_jits"] = []

    manifest = lean_path.parent / "jit-correspondence.json"
    records, pairs = [], []
    if manifest.is_file():
        manifest_name = manifest.relative_to(root).as_posix()
        hashes[manifest_name] = digest(manifest)
        records = json.loads(manifest.read_text())
        if not isinstance(records, list) or not records:
            raise ValueError(f"empty or invalid correspondence manifest: {manifest_name}")
        keys = [(r["python"], r["python_function"]) for r in records]
        lean_keys = [(r["lean"], r["lean_definition"]) for r in records]
        if len(keys) != len(set(keys)) or len(lean_keys) != len(set(lean_keys)):
            raise ValueError(f"reused Python or Lean correspondence entry: {manifest_name}")
        for record in records:
            for key in ("python", "lean"):
                path = local(root, record[key])
                hashes[record[key]] = digest(path)
        # Errors remain errors even for an incomplete manifest. They must not
        # be converted into a pending result that can pass the drift gate.
        results = check_manifest(root, manifest)
        for record, result in zip(records, results, strict=True):
            pairs.append({**result, "python": record["python"], "lean": record["lean"]})

    matched = {(p["python"], p["python_function"]): p for p in pairs}
    missing = [{"python": p, "function": fn} for p, fn in expected if (p, fn) not in matched]
    row["checked_pairs"] = pairs
    row["uncovered_selected_jits"] = missing
    python_casts = any(fn["explicit_int64_casts"] for fn in row["python_jits"])
    lean_casts = bool(re.search(r"\.to\s*\(\s*tl\.int64\b|tl\.cast\([^\n]*,\s*tl\.int64\b", code))
    row["int64_screen"] = ("cast_not_visible_in_primary_quotation" if python_casts
                            and row["lean_screen"]["quotations"] and not lean_casts else "inconclusive")
    row["body_copy_scope"] = "selected Python file and every explicitly declared correspondence pair"
    if python is None:
        row["status"] = "no_independent_python_reference"
    elif not expected:
        row["status"] = "no_selected_jit_resolved"
    elif missing:
        row["status"] = "uncovered_jit_functions"
    elif any(p["mode"] != "pasted_body" for p in pairs):
        row["status"] = "legacy_ast_only"
    else:
        row["status"] = ("authored_counterpart_body_copy_checked" if provenance == "authored_native_counterpart"
                         else "selected_body_copy_checked")
    row["source_sha256"] = dict(sorted(hashes.items()))
    return row


def scan(root: Path = ROOT) -> dict:
    root = root.resolve()
    ledger = json.loads((root / "bench/tritonbench_g/task_progress.json").read_text())
    tasks = []
    names = set()
    for entry in ledger["tasks"]:
        if entry["task"] in names:
            raise ValueError(f"duplicate selected task: {entry['task']}")
        names.add(entry["task"])
        revision = entry["kernel_revision"]
        tasks.append(inspect(root, entry["task"], revision["source"], revision["kernel_source"], "task"))
    examples = []
    references_path = root / "bench/examples/source-references.json"
    references = json.loads(references_path.read_text()) if references_path.is_file() else {}
    groups = {p.parent.name for p in (root / "bench/examples").glob("*/Kernels.lean")}
    if not isinstance(references, dict) or references.keys() - groups:
        raise ValueError("invalid or unknown worked-example source reference")
    for kernel in sorted((root / "bench/examples").glob("*/Kernels.lean")):
        reference = references.get(kernel.parent.name)
        provenance = "no_registered_reference"
        python = reference
        if isinstance(reference, dict):
            if reference.get("provenance") != "authored_native_counterpart" or not reference.get("note"):
                raise ValueError("authored example counterpart requires explicit provenance and explanation")
            python = reference.get("python")
            provenance = "authored_native_counterpart"
        elif isinstance(reference, str):
            provenance = "registered_python_reference"
        if python is not None and not isinstance(python, str):
            raise ValueError("worked-example source reference must name a reviewed Python file")
        examples.append(inspect(root, kernel.parent.name, python, kernel.relative_to(root).as_posix(), "example", provenance))
    counts = {"tasks": len(tasks), "examples": len(examples)}
    for kind, rows in (("task_status", tasks), ("example_status", examples)):
        counts[kind] = dict(sorted(Counter(r["status"] for r in rows).items()))
    counts["selected_jit_functions"] = sum(len(r["python_jits"]) for r in tasks)
    counts["uncovered_selected_jit_functions"] = sum(len(r["uncovered_selected_jits"]) for r in tasks)
    counts["primary_files_with_int64_screen_warning"] = sum(
        r["int64_screen"] == "cast_not_visible_in_primary_quotation" for r in tasks)
    counts["primary_lean_file_features"] = {
        kind: {feature: sum(bool(r["lean_screen"][feature]) for r in rows)
               for feature in ("in_body_type_directives", "antiquotes", "dsl_loop", "dsl_numeric_helpers")}
        for kind, rows in (("tasks", tasks), ("examples", examples))}
    return {"schema": 1,
            "scope": "All task_progress selected Python files and all bench/examples/*/Kernels.lean. Source correspondence only; Lean proof replay, proof reachability, hardware semantics and undeclared candidate Python files are not certified by this gate.",
            "complete": all(r["status"] in ("selected_body_copy_checked", "authored_counterpart_body_copy_checked")
                            for r in tasks + examples),
            "counts": counts, "tasks": tasks, "examples": examples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="record the full current inventory, including gaps")
    mode.add_argument("--check", action="store_true", help="reject inventory/source drift; pending remains pending")
    parser.add_argument("--require-complete", action="store_true", help="fail if any task or example is pending")
    args = parser.parse_args()
    try:
        result = scan(args.root)
        report = args.root / args.report
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.write:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(rendered)
        elif args.check:
            if not report.is_file() or json.loads(report.read_text()) != result:
                print("Source inventory drift: inspect changed pairs, replay affected proofs, then regenerate the report.", file=sys.stderr)
                return 1
        print(json.dumps({"complete": result["complete"], **result["counts"]}, ensure_ascii=False, indent=2))
        if args.require_complete and not result["complete"]:
            print("Body-copy acceptance FAILED: pending entries remain. An unchanged inventory is not a completion certificate.", file=sys.stderr)
            return 1
        return 0
    except (ValueError, KeyError, OSError, SyntaxError) as error:
        print(f"Body source check FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
