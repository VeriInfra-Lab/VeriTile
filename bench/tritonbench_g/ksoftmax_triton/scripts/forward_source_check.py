#!/usr/bin/env python3
"""Check the complete original forward softmax source AST for 216 dtype triples.

This dedicated checker retains the reviewed three Option String guard
restorations and destination-store conversion normalization. It does not add
an exemption to the shared audit, compile Lean profiles, or establish native
execution admission. Run from a VeriTile checkout or install under its bench
tree. The Python benchmark is parsed as text; it is never imported or launched.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile


BENCH_RELATIVE = Path("bench/tritonbench_g/ksoftmax_triton")
LEAN_RELATIVE = BENCH_RELATIVE / "Proofs/ForwardSource.lean"
PYTHON_SHA256 = "8f392086c34ec0cd665c31e614d1534fa2899fba0beecc6435cc06e2095d0e66"
FLOAT_TYPES = ("float32", "float64", "float16", "bfloat16", "float8e4nv", "float8e5")
BINDINGS = {
    "HAS_MASK": ("MASK_TYPE.isSome", "MASK_TYPE is not None"),
    "MASK_QK": ('MASK_TYPE == some "qk"', 'MASK_TYPE == "qk"'),
    "MASK_BK": ('MASK_TYPE == some "bk"', 'MASK_TYPE == "bk"'),
}
MUTATIONS = (
    ("if HAS_MASK", "if MASK_QK"),
    ("elif MASK_BK", "else"),
    ("M + n * $(stride_m)", "M + m * $(stride_m)"),
    ("k > n", "k >= n"),
    ("x - tl.max(x, axis=0)", "x + tl.max(x, axis=0)"),
    ("z - tl.log(denom)", "z + tl.log(denom)"),
    ("mask=k < $(K)", "mask=io_mask"),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repository_root(lean_source: Path | None) -> Path:
    anchors = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    if lean_source is not None:
        anchors.append(lean_source.resolve().parent)
    for anchor in anchors:
        for candidate in (anchor, *anchor.parents):
            if ((candidate / "bench/audit_source.py").is_file()
                    and (candidate / BENCH_RELATIVE / "ksoftmax_triton.py").is_file()):
                return candidate
    raise ValueError("Cannot find the VeriTile checkout; run from its tree")


def brace_body(text: str) -> ast.Module:
    """Reviewed brace-to-indentation conversion; unsupported syntax fails."""
    lines, current, parens = [], [], []
    depth = 0

    def flush(block: bool = False) -> None:
        line = "".join(current).strip()
        current.clear()
        if not line:
            return
        if block:
            if not (re.fullmatch(r"(?:if|elif)\s+.+", line, re.S) or line == "else"):
                raise ValueError("Unsupported block header: " + line)
            line += ":"
        lines.append("    " * depth + line)

    for char in text:
        if char in "([":
            parens.append(char)
            current.append(char)
        elif char in ")]":
            if not parens or parens.pop() != ("(" if char == ")" else "["):
                raise ValueError("Delimiter")
            current.append(char)
        elif char == "{":
            if parens:
                raise ValueError("Expression brace")
            flush(True)
            depth += 1
        elif char == "}":
            if parens or depth == 0:
                raise ValueError("Closing brace")
            flush()
            depth -= 1
        elif char == "\n" and not parens:
            flush()
        else:
            current.append(char)
    if parens or depth:
        raise ValueError("Unclosed source")
    flush()
    return ast.parse("\n".join(lines))


class Conditions(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in BINDINGS:
            if not isinstance(node.ctx, ast.Load):
                raise ValueError("Rebound guard")
            return ast.parse(BINDINGS[node.id][1], mode="eval").body
        return node


def checked_guard_view(text: str) -> str:
    """Restore only these three checked host bindings, never general lets."""
    audit_text = text
    for name, (lean, _) in BINDINGS.items():
        binding = "let " + name + " : Bool := " + lean
        if text.count(binding) != 1:
            raise ValueError("Expected exactly one checked binding: " + binding)
        # Preserve the reviewed literal binding restoration. These are host
        # conditions, not Lean macro-time source-splice definitions.
        audit_text = audit_text.replace(binding, "", 1)
    return audit_text


def check_body(body: str, original: str, params: set[str], source) -> None:
    actual = source.strip_region_declarations(body)

    def direct(match: re.Match[str]) -> str:
        if match[1] not in params:
            raise ValueError("Nonparameter antiquotation")
        return match[1]

    actual = re.sub(r"\$\((\w+)\)", direct, actual)
    if "$" in actual:
        raise ValueError("Unresolved splice")
    expected = ast.parse(original)
    stores = source.store_calls_with_explicit_destination_casts(original, body)
    if len(stores) != 1:
        raise ValueError("Expected exactly one destination store")
    expected.body[-1] = ast.parse(stores[0]).body[0]
    actual = Conditions().visit(brace_body(actual))
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        raise ValueError("Complete AST differs\n" + ast.unparse(ast.fix_missing_locations(actual)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-source", type=Path,
                        help="Lean source (default: checkout/" + str(LEAN_RELATIVE) + ")")
    parser.add_argument("--output-dir", type=Path,
                        help="New receipt directory (default: a fresh temporary directory)")
    args = parser.parse_args()
    root = repository_root(args.lean_source)
    lean_path = (args.lean_source if args.lean_source is not None
                 else root / LEAN_RELATIVE).resolve()
    paths = {
        "python_source": (root / BENCH_RELATIVE / "ksoftmax_triton.py").resolve(),
        "lean_source": lean_path,
        "checker": Path(__file__).resolve(),
        "audit_helper": (root / "bench/audit_source.py").resolve(),
    }
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    before = {name: digest(raw) for name, raw in inputs.items()}
    if before["python_source"] != PYTHON_SHA256:
        raise ValueError("Pinned forward Python source hash mismatch")

    sys.path.insert(0, str(root))
    from bench import audit_source as source
    if Path(source.__file__).resolve() != paths["audit_helper"]:
        raise ValueError("Unexpected shared source audit helper")

    text = inputs["lean_source"].decode("utf-8")
    tree = ast.parse(inputs["python_source"])
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_softmax"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one original _softmax function")
    fn = functions[0]
    if not (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        raise ValueError("Expected the pinned original function docstring")
    params = {arg.arg for arg in fn.args.args}
    original = ast.unparse(ast.Module(body=fn.body[1:], type_ignores=[]))
    audit_text = checked_guard_view(text)
    rows = []
    for x, m, y in itertools.product(FLOAT_TYPES, repeat=3):
        metadata = ("/-\nSource-audit Lean: forwardSource%\nSource-audit args: "
                    f"inputType=tl.{x}, maskType=tl.{m}, destinationType=tl.{y}, "
                    "version=original\n-/\n")
        body = source.lean_first_triton_body(
            source.select_lean_audit_source(audit_text, metadata))
        if source.declared_region_dtypes(body) != {
                "X": {"tl." + x}, "M": {"tl." + m}, "Y": {"tl." + y}}:
            raise ValueError(f"Region dtype mismatch: {x}/{m}/{y}")
        check_body(body, original, params, source)
        for old, new in MUTATIONS:
            if old not in body:
                raise ValueError("Mutation anchor missing: " + old)
            try:
                check_body(body.replace(old, new, 1), original, params, source)
            except (ValueError, SyntaxError):
                pass
            else:
                raise ValueError("Accepted mutation: " + old)
        rows.append({
            "input": x, "mask": m, "destination": y,
            "full_ast_match": True, "negative_controls": len(MUTATIONS),
            "body_sha256": digest(body.encode()),
        })
    profiles = len(rows)
    controls = sum(row["negative_controls"] for row in rows)
    if profiles != 216 or controls != 1512:
        raise ValueError("Incomplete profile or mutation matrix")
    after = {name: digest(path.read_bytes()) for name, path in paths.items()}
    if before != after:
        changed = [name for name in paths if before[name] != after[name]]
        raise ValueError("Inputs changed during source comparison: " + ", ".join(changed))

    if args.output_dir is None:
        out = Path(tempfile.mkdtemp(prefix="ksoftmax-forward-source-"))
    else:
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "profiles": rows,
        "profile_count": profiles,
        "negative_control_count": controls,
        "source_sha256": before["lean_source"],
        "python_source_sha256": before["python_source"],
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_hashes_before": before,
        "input_hashes_after": after,
        "inputs_unchanged": True,
        "guard_bindings": {name: {"lean": lean, "python": python}
                           for name, (lean, python) in BINDINGS.items()},
        "mutations": [{"before": old, "after": new} for old, new in MUTATIONS],
        "scope": "Dedicated complete original forward-source AST comparison after the "
                 "three reviewed Option String guard restorations and destination-store "
                 "conversion. All 216 dtype triples and 1512 negative controls are source "
                 "checks only; no generic audit integration, all-profile Lean compilation, "
                 "typed successful execution, native admission or proof completion is claimed.",
        "complete_task": False,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiles": profiles, "negative_controls": controls,
                      "inputs_unchanged": True, "receipt": str(out / "receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
