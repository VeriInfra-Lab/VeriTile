#!/usr/bin/env python3
"""Inventory source correspondence for every task directory under bench.

This is a source gate, not a Lean proof or a compiler-correctness certificate.
`--check` checks the recorded inventory, including pending entries, for drift.
`--require-complete` additionally rejects ANY pending entry. A partial manifest
cannot certify a multi-stage Python file. No variable renaming, cast deletion,
or generated Python reference is used to manufacture a correspondence pass.
"""
from __future__ import annotations

# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from scripts.bench.audit_source import lean_code_mask
from scripts.bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("validation/source-correspondence.json")


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


def inspect(root: Path, task: str, python: str | None, lean: str, kind: str) -> dict:
    row = {"name": task, "kind": kind, "python": python, "lean": lean}
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
        if len(keys) != len(set(keys)):
            raise ValueError(f"reused Python correspondence entry: {manifest_name}")
        # Distinct original/candidate files may legitimately share an unchanged
        # stage. Check every Python body against that Lean definition instead
        # of forcing a duplicate Lean implementation just to register it.
        for record in records:
            for key in ("python", "lean"):
                path = local(root, record[key])
                hashes[record[key]] = digest(path)
        # Errors remain errors even for an incomplete manifest. They must not
        # be converted into a pending result that can pass the drift gate.
        results = check_manifest(root, manifest)
        for record, result in zip(records, results, strict=True):
            pairs.append({**result, "python": record["python"], "lean": record["lean"]})

    # Inventory all retained task-local Python sources, including dynamically
    # loaded bodies and unregistered candidates. Merely checking the selected
    # wrapper or the files already named by a manifest can miss entire stages.
    sources = {p.relative_to(root).as_posix() for p in lean_path.parent.rglob("*.py")
               if "__pycache__" not in p.parts}
    if python is not None:
        sources.add(python)
    sources.update(record["python"] for record in records)
    row["python_sources"] = []
    for name in sorted(sources):
        path = local(root, name)
        functions = jit_functions(path)
        names = [fn["function"] for fn in functions]
        if len(names) != len(set(names)):
            raise ValueError(f"ambiguous Python JIT names: {name}")
        hashes[name] = digest(path)
        row["python_sources"].append({"path": name, "jits": functions})
    expected = [(source["path"], fn["function"])
                for source in row["python_sources"] for fn in source["jits"]]

    matched = {(p["python"], p["python_function"]): p for p in pairs}
    missing = [{"python": p, "function": fn} for p, fn in expected if (p, fn) not in matched]
    row["checked_pairs"] = pairs
    row["uncovered_python_jits"] = missing
    row["uncovered_selected_jits"] = [entry for entry in missing if entry["python"] == python]
    python_casts = any(fn["explicit_int64_casts"] for fn in row["python_jits"])
    lean_casts = bool(re.search(r"\.to\s*\(\s*tl\.int64\b|tl\.cast\([^\n]*,\s*tl\.int64\b", code))
    row["int64_screen"] = ("cast_not_visible_in_primary_quotation" if python_casts
                            and row["lean_screen"]["quotations"] and not lean_casts else "inconclusive")
    row["body_copy_scope"] = "selected reference, all task-local Python files and all manifest pairs; no proof or backend certification"
    if python is None:
        row["status"] = "no_independent_python_reference"
    elif not row["python_jits"]:
        row["status"] = "no_selected_jit_resolved"
    elif missing:
        row["status"] = "uncovered_jit_functions"
    elif any(p["mode"] != "pasted_body" for p in pairs):
        row["status"] = "legacy_ast_only"
    else:
        row["status"] = "selected_body_copy_checked"
    row["source_sha256"] = dict(sorted(hashes.items()))
    return row


def task_directories(root: Path) -> list[tuple[str, Path]]:
    """Discover tasks from disk, never from the selected-proof ledger.

    Collections use either <collection>/<task> or <collection>/tasks/<task>.
    Keep incomplete directories visible: a missing Kernels.lean is an error,
    rather than a reason to silently remove that problem from the inventory.
    """
    bench = root / "bench"
    if not bench.is_dir():
        raise ValueError("missing bench directory")
    result = []
    for collection in sorted(bench.iterdir()):
        if not collection.is_dir() or collection.name.startswith((".", "_")):
            continue
        base = collection / "tasks" if (collection / "tasks").is_dir() else collection
        for directory in sorted(base.iterdir()):
            if directory.is_dir() and not directory.name.startswith((".", "_")):
                result.append((collection.name, directory))
    return result


def scan(root: Path = ROOT) -> dict:
    root = root.resolve()
    ledger_path = root / "validation/task_progress.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {"tasks": []}
    selected = {}
    for entry in ledger["tasks"]:
        if entry["task"] in selected:
            raise ValueError(f"duplicate selected task: {entry['task']}")
        selected[entry["task"]] = entry["kernel_revision"]
    directories = task_directories(root)
    references = {}
    for collection in sorted({name for name, _ in directories}):
        path = root / "bench" / collection / "source-references.json"
        mapping = json.loads(path.read_text()) if path.is_file() else {}
        known = {p.name for name, p in directories if name == collection}
        if not isinstance(mapping, dict) or mapping.keys() - known:
            raise ValueError(f"invalid or unknown {collection} source reference")
        if any(not isinstance(value, str) for value in mapping.values()):
            raise ValueError("source reference must name a reviewed Python file")
        references[collection] = mapping
    tasks, examples = [], []
    discovered_selected = set()
    for collection, directory in directories:
        kernel = directory / "Kernels.lean"
        python = references[collection].get(directory.name)
        if collection == "tritonbench_g" and directory.name in selected:
            revision = selected[directory.name]
            discovered_selected.add(directory.name)
            if local(root, revision["kernel_source"]) != kernel:
                raise ValueError(f"selected kernel is not the task entry: {directory.name}")
            if python is None:
                python = revision["source"]
        elif python is None and (directory / (directory.name + ".py")).is_file():
            python = (directory / (directory.name + ".py")).relative_to(root).as_posix()
        kind = "example" if collection == "examples" else "task"
        row = inspect(root, directory.name, python, kernel.relative_to(root).as_posix(), kind)
        row.update(collection=collection, id=directory.relative_to(root).as_posix())
        metadata = directory / "task.json"
        if metadata.is_file():
            row["source_sha256"][metadata.relative_to(root).as_posix()] = digest(metadata)
        (examples if kind == "example" else tasks).append(row)
    if set(selected) - discovered_selected:
        raise ValueError(f"selected tasks missing from bench: {sorted(set(selected) - discovered_selected)}")
    all_rows = tasks + examples
    counts = {"total": len(all_rows), "tasks": len(tasks), "examples": len(examples),
              "collections": dict(sorted(Counter(r["collection"] for r in all_rows).items()))}
    for kind, rows in (("task_status", tasks), ("example_status", examples)):
        counts[kind] = dict(sorted(Counter(r["status"] for r in rows).items()))
    counts["selected_jit_functions"] = sum(len(r["python_jits"]) for r in tasks)
    counts["uncovered_selected_jit_functions"] = sum(len(r["uncovered_selected_jits"]) for r in tasks)
    counts["inventoried_jit_functions"] = sum(len(source["jits"]) for row in all_rows
                                             for source in row["python_sources"])
    counts["uncovered_inventoried_jit_functions"] = sum(len(r["uncovered_python_jits"]) for r in all_rows)
    counts["checked_pair_modes"] = dict(sorted(Counter(pair["mode"] for row in all_rows
                                                       for pair in row["checked_pairs"]).items()))
    counts["collection_status"] = {
        collection: dict(sorted(Counter(r["status"] for r in all_rows
                                       if r["collection"] == collection).items()))
        for collection in counts["collections"]}
    counts["primary_files_with_int64_screen_warning"] = sum(
        r["int64_screen"] == "cast_not_visible_in_primary_quotation" for r in tasks)
    counts["primary_lean_file_features"] = {
        kind: {feature: sum(bool(r["lean_screen"][feature]) for r in rows)
               for feature in ("in_body_type_directives", "antiquotes", "dsl_loop", "dsl_numeric_helpers")}
        for kind, rows in (("tasks", tasks), ("examples", examples))}
    return {"schema": 2,
            "scope": "Every bench task directory, including imported challenges; selected references, all task-local Python files and manifest pairs. Missing references remain pending. No Lean proof, semantic-connection or hardware certification.",
            "complete": bool(all_rows) and all(r["status"] == "selected_body_copy_checked" for r in all_rows),
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
