"""Frozen task-local transcription drift guard; this is not a compiler proof."""
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "parallel_attention_repaired.py"
KERNELS = (
    "parallel_rebased_fwd_kernel",
    "_parallel_rebased_bwd_dq",
    "_parallel_rebased_bwd_dkv",
    "parallel_rebased_bwd_kernel",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    pins = json.loads((HERE / "source-pins.json").read_text())
    nodes = {node.name: node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef)}
    actual = {name: hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
              for name in KERNELS}
    retained = json.loads((HERE.parent / "checks" / "kernel-ast.json").read_text())
    assert actual == retained == pins["python_kernel_asts"], "Python kernel AST drift"
    for name, expected in pins["lean_transcriptions"].items():
        assert digest(HERE / name) == expected, f"Lean transcription drift: {name}"
    quotes = "\n".join((HERE / name).read_text() for name in ("SourceSyntax.lean", "QuerySyntax.lean"))
    counts = {}
    for op in ("dot", "load", "store"):
        python_count = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                           and isinstance(n.func.value, ast.Name) and n.func.value.id == "tl"
                           and n.func.attr == op for name in KERNELS for n in ast.walk(nodes[name]))
        lean_count = len(re.findall(r"\btl\." + op + r"\(", quotes))
        assert python_count == lean_count, f"Source operation count drift: {op}"
        counts[op] = python_count
    assert counts == {"dot": 18, "load": 20, "store": 5}, counts
    return {"status": "passed", "scope": "frozen transcription drift and structural counts",
            "semantic_equivalence_by_hash": False, "counts": counts,
            "source_sha256": digest(SOURCE), "pins_sha256": digest(HERE / "source-pins.json")}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
