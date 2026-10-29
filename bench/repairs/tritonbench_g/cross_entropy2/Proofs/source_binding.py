"""Source pins and unchanged JIT/host AST; no host execution or GPU claim."""
import ast
import hashlib
import json
from pathlib import Path
D = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checked = D / "cross_entropy2_repaired.py"
conditional = D / "cross_entropy2_conditional.py"
assert sha(checked) == "628b6d679381ed6d1353fc070a313cc7587ca52676a84eacf16637323c575ba9"
assert sha(conditional) == "741315fb177ec4cf0c7325c8244c87a8cbc95921511d81e41468d304ce5e8b6a"
def asts(path, strip):
    nodes = []
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.FunctionDef) and (n.name.endswith("_kernel") or n.name in ("cross_entropy_fwd", "cross_entropy_bwd")):
            if strip:
                class Checks(ast.NodeTransformer):
                    def visit_Expr(self, node):
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_check_row": return None
                        return self.generic_visit(node)
                    def visit_Name(self, node):
                        if node.id == "_checked_validate": node.id = "_validate"
                        return node
                n = Checks().visit(n)
            nodes.append(ast.dump(n, include_attributes=False))
    return nodes
assert asts(checked, True) == asts(conditional, False)
print(json.dumps({"selected_source_sha256": sha(checked), "conditional_source_sha256": sha(conditional),
    "unchanged_kernel_and_host_AST": True, "gpu_execution": False}, indent=2))
