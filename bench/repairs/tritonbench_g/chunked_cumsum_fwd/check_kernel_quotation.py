"""Compare the complete readable Lean quotation with the selected Python AST.

This audit removes only region-type metadata and Lean antiquotation brackets.
It does not execute a second kernel, normalize arithmetic, remove casts, or
change a branch, mask, memory operation, or scalar parameter.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    python = here / "chunked_cumsum_fwd_repaired.py"
    lean = here / "Proofs/KernelSource.lean"
    source = ast.parse(python.read_text())
    original = next(x for x in source.body if isinstance(x, ast.FunctionDef)
                    and x.name == "_chunk_cumsum_fwd_kernel")
    text = lean.read_text()
    body = text.split("  chunked_runtime_source {\n", 1)[1].split("\n  }", 1)[0]
    body = body[body.index("    pid_b ="):]
    body = re.sub(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", r"\1", body)
    lines, depth = [], 0
    for line in body.splitlines():
        line = line.strip()
        if line == "}":
            depth -= 1
            assert depth >= 0
        elif line.endswith("{"):
            assert line.startswith("if ")
            lines.append("    " * depth + line[:-1].rstrip() + ":")
            depth += 1
        else:
            lines.append("    " * depth + line)
    assert depth == 0
    quoted = ast.parse("\n".join(lines) + "\n")
    expected = ast.Module(body=original.body, type_ignores=[])
    assert ast.dump(quoted, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert '[("dt_min",.real,SInfo.scalar,none),' in text
    assert '("dt_max",.real,SInfo.scalar,none)]' in text
    assert "let dtype : FloatDType := .fp32" in text
    assert "let out_dtype : FloatDType := .fp32" in text
    assert "tl.region dt_ptr = DTYPE, A_ptr = OUT_DTYPE, dt_bias_ptr = tl.float32" in text
    assert "dt_out_ptr = tl.float32, dA_cumsum_ptr = tl.float32" in text
    report = {"status": "passed", "python": str(python), "python_sha256": digest(python),
              "lean": str(lean), "lean_sha256": digest(lean),
              "top_level_statements": len(original.body),
              "all_statements": sum(isinstance(x, ast.stmt) for x in ast.walk(expected)),
              "complete_ast_equal": True, "runtime_bounds": ["dt_min", "dt_max"],
              "normalization": "region dtype metadata, Lean antiquotation brackets and block braces only",
              "scope": "Quotation/source identity; Lean execution and native lowering are separate evidence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
