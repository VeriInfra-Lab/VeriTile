"""Check the complete Python candidate differs only at integer matmul group width."""
import ast
from pathlib import Path

D = Path(__file__).resolve().parent
original = ast.parse((D.parent / "triton_matmul_repaired.py").read_text())
candidate_path = D / "triton_matmul_shifted.py"
candidate = ast.parse(candidate_path.read_text())
compile(candidate, str(candidate_path), "exec")
dump = lambda node: ast.dump(node, include_attributes=False)
kernel = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == "matmul_kernel_shifted")
sites = [n for n in kernel.body if isinstance(n, ast.If) and dump(n.test) == dump(ast.parse("GROUP_SIZE_M == 8", mode="eval").body)]
assert len(sites) == 1
site = sites[0]
assert len(site.body) == len(site.orelse) == 1
assert dump(site.body[0]) == dump(ast.parse("num_pid_in_group = num_pid_n << 3").body[0])
assert dump(site.orelse[0]) == dump(ast.parse("num_pid_in_group = GROUP_SIZE_M * num_pid_n").body[0])
kernel.body[kernel.body.index(site)] = site.orelse[0]
names = {"matmul_kernel_shifted": "matmul_kernel_repaired", "matmul_shifted": "matmul_repaired"}
class RestoreNames(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = names.get(node.id, node.id)
        return node
    def visit_FunctionDef(self, node):
        node.name = names.get(node.name, node.name)
        return self.generic_visit(node)
candidate = RestoreNames().visit(candidate)
assert dump(original) == dump(candidate), "unexpected full-module candidate delta"
print("PASS: exact full-module AST restoration; GROUP_SIZE_M=8 shift and generic fallback; both storage configs, branches, helpers and host preserved")
