#!/usr/bin/env python3
"""Inventory body-copy readiness for every selected task and worked example.

This is a source gate, not a Lean proof or a compiler-correctness certificate.
`--check` checks the recorded inventory, including pending entries, for drift.
`--require-complete` additionally rejects ANY pending entry. A partial manifest
cannot certify a multi-stage Python file. No variable renaming, cast deletion,
or generated Python mirror certifies a selected benchmark's original source.
New native counterparts and sources authored from independent mathematical
specifications have separate provenance from historical Python references.
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
from bench.jit_correspondence import check_manifest, validate_manifest_records

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


def delegated_sources(root: Path, wrapper: Path) -> list[dict]:
    """Recognize the explicit local importlib loader used by checked wrappers.

    Read source only: no imports or wrapper code are executed. Resolve only a
    literal sibling filename, with the standard pathlib/importlib bindings and
    one unmodified spec/module binding. Every JIT in the loaded file is required,
    including functions selected by host branches inside that file.
    """
    tree = ast.parse(wrapper.read_text())
    standard_importlib = any(isinstance(n, ast.Import) and any(
        a.name == "importlib.util" and a.asname is None for a in n.names) for n in tree.body)
    standard_path = any(isinstance(n, ast.ImportFrom) and n.module == "pathlib"
        and n.level == 0 and any(a.name == "Path" and a.asname is None for a in n.names) for n in tree.body)
    if not (standard_importlib and standard_path):
        return []
    specs, modules, resolved = {}, {}, []
    available = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            available.update(a.name for a in node.names if a.asname is None)
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib" and node.level == 0:
            available.update("pathlib." + a.name for a in node.names if a.asname is None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
            if not isinstance(value, ast.Call):
                continue
            if (ast.unparse(value.func) == "importlib.util.spec_from_file_location"
                    and len(value.args) == 2 and not value.keywords
                    and {"importlib.util", "pathlib.Path"} <= available):
                target = value.args[1]
                if not (isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str)
                        and isinstance(target, ast.Call) and isinstance(target.func, ast.Attribute)
                        and target.func.attr == "with_name" and len(target.args) == 1
                        and not target.keywords and ast.unparse(target.func.value) == "Path(__file__)"
                        and isinstance(target.args[0], ast.Constant) and isinstance(target.args[0].value, str)):
                    continue
                filename = target.args[0].value
                if Path(filename).name != filename or not filename.endswith(".py"):
                    raise ValueError("delegated Python source must be a literal sibling .py file")
                path = local(root, (wrapper.parent / filename).relative_to(root).as_posix())
                if not path.is_file():
                    raise ValueError(f"missing delegated Python source: {path}")
                specs[name] = (path, node.lineno)
            elif (ast.unparse(value.func) == "importlib.util.module_from_spec"
                    and len(value.args) == 1 and not value.keywords
                    and isinstance(value.args[0], ast.Name) and value.args[0].id in specs):
                modules[name] = value.args[0].id
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Name):
                continue
            module = call.args[0].id
            if module not in modules:
                continue
            spec = modules[module]
            if ast.unparse(call.func) == spec + ".loader.exec_module":
                path, line = specs[spec]
                resolved.append({"python": path.relative_to(root).as_posix(), "spec": spec,
                    "module_binding": module, "spec_line": line, "load_line": node.lineno})
    if not resolved:
        return []
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
           for n in ast.walk(tree)):
        raise ValueError("wildcard import makes delegated loader binding ambiguous")
    protected = {"importlib": 0, "Path": 0, "__file__": 0}
    for entry in resolved:
        protected[entry["spec"]] = protected[entry["module_binding"]] = 1
    stores = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            stores[node.id] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores[node.name] += 1
        elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            value = node.value
            while isinstance(value, (ast.Attribute, ast.Subscript)):
                value = value.value
            if isinstance(value, ast.Name) and value.id in protected:
                raise ValueError("delegated Python loader binding is mutated")
    if any(stores[name] != count for name, count in protected.items()):
        raise ValueError("delegated Python loader binding is rebound")
    # A second import of a protected name is also a rebinding, even though
    # Python's AST does not represent import targets with ast.Store.
    imported = Counter(a.asname or (a.name.split(".")[0] if isinstance(n, ast.Import) else a.name)
        for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names)
    if imported["importlib"] != 1 or imported["Path"] != 1 or any(
            imported[name] for name in protected if name not in {"importlib", "Path"}):
        raise ValueError("delegated Python loader import is rebound")
    return resolved


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
        if not expected:
            delegation = delegated_sources(root, path)
            if delegation:
                row["python_source_delegations"] = delegation
                for source in dict.fromkeys(entry["python"] for entry in delegation):
                    source_path = local(root, source)
                    hashes[source] = digest(source_path)
                    functions = jit_functions(source_path)
                    resolved_names = [fn["function"] for fn in functions]
                    if len(resolved_names) != len(set(resolved_names)):
                        raise ValueError(f"ambiguous delegated Python JIT names: {source}")
                    row["python_jits"].extend({**fn, "python": source} for fn in functions)
                    expected.extend((source, name) for name in resolved_names)
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
        validate_manifest_records(records)
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
    if row.get("python_source_delegations"):
        row["body_copy_scope"] += "; every JIT in the statically resolved local delegated source"
    if python is None:
        row["status"] = "no_independent_python_reference"
    elif not expected:
        row["status"] = "no_selected_jit_resolved"
    elif missing:
        row["status"] = "uncovered_jit_functions"
    elif any(p["mode"] != "pasted_body" for p in pairs):
        row["status"] = "legacy_ast_only"
    else:
        row["status"] = {"authored_native_counterpart": "authored_counterpart_body_copy_checked",
                         "spec_derived_native_source": "spec_derived_body_copy_checked"}.get(
                             provenance, "selected_body_copy_checked")
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
            if reference.get("provenance") not in ("authored_native_counterpart", "spec_derived_native_source") or not reference.get("note"):
                raise ValueError("authored example source requires explicit provenance and explanation")
            python = reference.get("python")
            provenance = reference["provenance"]
            if provenance == "spec_derived_native_source":
                pins = reference.get("mathematical_spec_sha256")
                if not isinstance(pins, dict) or not pins:
                    raise ValueError("spec-derived source requires pinned independent mathematical specifications")
                for name, expected in pins.items():
                    path = local(root, name)
                    if name in (python, kernel.relative_to(root).as_posix()) or path.suffix != ".lean":
                        raise ValueError("mathematical specification cannot be the translated kernel or Python source")
                    if not path.is_file() or digest(path) != expected:
                        raise ValueError(f"mathematical specification pin drift: {name}")
        elif isinstance(reference, str):
            provenance = "registered_python_reference"
        if python is not None and not isinstance(python, str):
            raise ValueError("worked-example source reference must name a reviewed Python file")
        row = inspect(root, kernel.parent.name, python, kernel.relative_to(root).as_posix(), "example", provenance)
        if isinstance(reference, dict):
            row["source_note"] = reference["note"]
            if provenance == "spec_derived_native_source":
                row["mathematical_spec_sha256"] = reference["mathematical_spec_sha256"]
                row["source_sha256"].update(reference["mathematical_spec_sha256"])
                row["source_sha256"] = dict(sorted(row["source_sha256"].items()))
        examples.append(row)
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
            "complete": all(r["status"] in ("selected_body_copy_checked", "authored_counterpart_body_copy_checked", "spec_derived_body_copy_checked")
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
