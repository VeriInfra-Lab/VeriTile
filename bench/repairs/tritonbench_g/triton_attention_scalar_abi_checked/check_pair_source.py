"""Verify the complete three-stage Python candidate's sole integer rewrite."""
import ast
from pathlib import Path
D = Path(__file__).resolve().parent
a = ast.parse((D / "triton_attention_repaired.py").read_text())
p = D / "triton_attention_shifted.py"
b = ast.parse(p.read_text())
compile(b, str(p), "exec")
dump = lambda n: ast.dump(n, include_attributes=False)
k = next(n for n in b.body if isinstance(n, ast.FunctionDef) and n.name == "_bwd_preprocess_shifted")
assert dump(k.body[0]) == dump(ast.parse("tl.static_assert(BLOCK_M == 128)").body[0])
k.body.pop(0)
assert dump(k.body[0]) == dump(ast.parse("off_m = (tl.program_id(0) << 7) + tl.arange(0, BLOCK_M)").body[0])
k.body[0] = ast.parse("off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)").body[0]
class Restore(ast.NodeTransformer):
    def visit_Name(self, n):
        if n.id == "_bwd_preprocess_shifted": n.id = "_bwd_preprocess"
        return n
    def visit_FunctionDef(self, n):
        if n.name == "_bwd_preprocess_shifted": n.name = "_bwd_preprocess"
        return self.generic_visit(n)
b = Restore().visit(b)
assert dump(a) == dump(b), "unexpected complete pipeline/host delta"
print("PASS: exact complete-module AST restoration; one fixed-128 preprocess shift and static assertion; forward/backward, scalar ABI, casts, grids, masks, host flow unchanged")
