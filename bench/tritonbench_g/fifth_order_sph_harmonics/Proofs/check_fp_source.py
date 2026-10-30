"""Compare the complete forward expression graph to the local Python source.
Only explicit dtype declarations and coefficient type annotations are erased.
This source check does not assert compiler or native execution equivalence.
"""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def check():
    py = ast.parse((ROOT / "fifth_order_sph_harmonics.py").read_text())
    forward = next(n for n in py.body if isinstance(n, ast.FunctionDef) and n.name == "fifth_order_fwd")
    source = (ROOT / "Proofs/TypedSource.lean").read_text().split("triton {", 1)[1].split("\n}", 1)[0]
    source = re.sub(r"\$\((\w+)\)", r"\1", source)
    parts = []
    for line in source.splitlines():
        if not line.strip() or line.strip().startswith("tl.region"):
            continue
        if re.match(r"^  (?:\w+ =|tl.store\()", line):
            parts.append(line.strip())
        else:
            parts[-1] += " " + line.strip()
    translated = ast.parse("\n".join(parts)).body
    for stmt in translated:
        if isinstance(stmt, ast.Assign) and stmt.targets[0].id.startswith("CONST"):
            assert isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == "tl.full"
            assert len(stmt.value.keywords) == 1 and ast.unparse(stmt.value.keywords[0].value) == "tl.float32"
            stmt.value = stmt.value.args[1]
    original = [n for n in forward.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
    assert len(original) == len(translated), (len(original), len(translated))
    for original_stmt, typed_stmt in zip(original, translated):
        assert ast.dump(original_stmt) == ast.dump(typed_stmt), (ast.unparse(original_stmt), ast.unparse(typed_stmt))
    return {"statements": len(original), "stores": sum(isinstance(n, ast.Expr) for n in original),
            "status": "PASS", "native_binding": "not_established"}

if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=2))
