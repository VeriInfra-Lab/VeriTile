"""Check complete argmax candidate preservation and its exact guarded shifts."""
import ast
from pathlib import Path
D = Path(__file__).resolve().parent
a = ast.parse((D / "triton_argmax_nan_first.py").read_text())
p = D / "triton_argmax_shifted.py"
b = ast.parse(p.read_text())
compile(b, str(p), "exec")
dump = lambda n: ast.dump(n, include_attributes=False)
for name, guard, shifted, original in [
    ("argmax_kernel_1_shifted", "BLOCK_SIZE == 1024",
     "offset = (pid.to(tl.int64) << 10) + tl.arange(0, BLOCK_SIZE).to(tl.int64)",
     "offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)"),
    ("argmax_kernel_shifted", "BLOCK_M == 128",
     "m_offset = (pid_m.to(tl.int64) << 7) + tl.arange(0, BLOCK_M).to(tl.int64)",
     "m_offset = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)"),
]:
    k = next(n for n in b.body if isinstance(n, ast.FunctionDef) and n.name == name)
    sites = [n for n in k.body if isinstance(n, ast.If) and dump(n.test) == dump(ast.parse(guard, mode="eval").body)]
    assert len(sites) == 1
    site = sites[0]
    assert len(site.body) == len(site.orelse) == 1
    assert dump(site.body[0]) == dump(ast.parse(shifted).body[0])
    assert dump(site.orelse[0]) == dump(ast.parse(original).body[0])
    k.body[k.body.index(site)] = site.orelse[0]
names = {"argmax_kernel_1_shifted": "argmax_kernel_1_repaired", "argmax_kernel_shifted": "argmax_kernel_repaired", "argmax_shifted": "argmax"}
class Restore(ast.NodeTransformer):
    def visit_Name(self, n):
        n.id = names.get(n.id, n.id)
        return n
    def visit_FunctionDef(self, n):
        n.name = names.get(n.name, n.name)
        return self.generic_visit(n)
b = Restore().visit(b)
assert dump(a) == dump(b), "unexpected whole-module source delta"
print("PASS: exact complete-module AST restoration; global1024 and dimension128 guarded shifts with generic fallbacks; all requested casts/formats, NaN flags, reductions, stages, casts, masks, host flow preserved")

# Both Python modules must also match the actual Lean source bodies.
import sys
ROOT = D.parents[3]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
pairs = check_manifest(ROOT, D / "jit-correspondence.json")
print(f"PASS: {len(pairs)} complete Python JIT / Lean quotation pairs")
