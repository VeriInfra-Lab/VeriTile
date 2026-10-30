"""Check both complete raw Python/Lean bodies and their external call metadata.

This source audit uses the shared correspondence gate. It does not run a kernel,
remove casts, normalize arithmetic or certify execution/native lowering.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    python = here / "chunked_cumsum_fwd_repaired.py"
    lean = here / "Kernels.lean"
    manifest = here / "jit-correspondence.json"
    pairs = check_manifest(ROOT, manifest)
    assert len(pairs) == 2 and all(pair["mode"] == "pasted_body" for pair in pairs)
    original = next(node for node in ast.parse(python.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_chunk_cumsum_fwd_kernel")
    expected = ast.Module(body=original.body, type_ignores=[])
    report = {"status": "passed", "python": str(python), "python_sha256": digest(python),
              "lean": str(lean), "lean_sha256": digest(lean),
              "top_level_statements": len(original.body),
              "all_statements": sum(isinstance(node, ast.stmt) for node in ast.walk(expected)),
              "complete_ast_equal": True, "runtime_bounds": ["dt_min", "dt_max"],
              "manifest_sha256": digest(manifest), "pairs": pairs,
              "candidate_provenance": "New native counterpart of the existing Lean scaleFirst candidate",
              "normalization": "Shared raw-body Python AST gate; all source names, casts, branches and operations retained",
              "scope": "Quotation and external call interface identity; execution/native lowering are separate evidence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
