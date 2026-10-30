"""Emit or check the source-only Lean encoding of the complete Python AST."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "layer_norm_triton_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/layer_norm_triton/layer_norm_triton.py"


def encode(value):
    if isinstance(value, ast.AST):
        fields = ",".join(f"({json.dumps(k)},{encode(v)})" for k, v in ast.iter_fields(value))
        return f'.node "{type(value).__name__}" [{fields}]'
    if isinstance(value, list):
        return ".nodes [" + ",".join(map(encode, value)) + "]"
    if value is None:
        return ".none"
    if isinstance(value, bool):
        return ".boolean " + ("true" if value else "false")
    if isinstance(value, str):
        return ".text " + json.dumps(value)
    if isinstance(value, (int, float)):
        return ".number " + json.dumps(repr(value))
    raise TypeError(type(value))


def generated():
    tree = ast.parse(SOURCE.read_text())
    header = '''import VeriTile.Triton

/-! Complete checked Python source AST. Numerical models, compiler bindings,
execution proofs and mathematical specifications live in Proofs. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Source
inductive Syntax where
  | node (tag : String) (fields : List (String × Syntax))
  | nodes (items : List Syntax)
  | text (value : String)
  | number (literal : String)
  | boolean (value : Bool)
  | none
  deriving Inhabited
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
'''
    pins = f'def originalSHA256 := "{hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()}"\ndef candidateSHA256 := "{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}"\n'
    declarations = "".join(f"def declaration{i} : Syntax := {encode(node)}\n" for i, node in enumerate(tree.body))
    module = 'def module : Syntax := .node "Module" [("body",.nodes [' + ",".join(f"declaration{i}" for i in range(len(tree.body))) + ']),("type_ignores",.nodes [])]\n'
    return header + pins + declarations + module + "end VeriTile.Bench.Repairs.LayerNormTriton.Source\n"


if __name__ == "__main__":
    target = HERE / "Proofs/PythonSource.lean"
    if "--check" in sys.argv:
        assert target.read_text() == generated(), "source-only AST or hashes are stale"
        print("Complete source AST and current hashes match.")
    else:
        target.write_text(generated())
