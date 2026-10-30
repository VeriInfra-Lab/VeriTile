"""Validate the exact fixed-tile optimization delta without importing Torch."""
import ast
from pathlib import Path
here = Path(__file__).resolve().parent
original = ast.parse((here/'Whole001'/'attention_fwd_triton2_repaired.py').read_text())
candidate = ast.parse((here/'attention_fwd_triton2_shifted.py').read_text())
function = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name=='_attn_fwd_shifted')
assert ast.unparse(function.body.pop(0)) == 'tl.static_assert(BLOCK_M == 128)'
function.name = '_attn_fwd'
class Restore(ast.NodeTransformer):
    count = 0
    launches = 0
    def visit_FunctionDef(self, node):
        if node.name == 'forward_shifted': node.name = 'forward'
        return self.generic_visit(node)
    def visit_Name(self, node):
        if node.id == '_attn_fwd_shifted':
            node.id = '_attn_fwd'
            self.launches += 1
        return node
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.LShift):
            assert ast.unparse(node) == 'start_m << 7'
            self.count += 1
            return ast.BinOp(ast.Name('start_m', ast.Load()), ast.Mult(), ast.Name('BLOCK_M', ast.Load()))
        return node
restore = Restore()
candidate = restore.visit(candidate)
assert restore.count == 1 and restore.launches == 1
assert ast.dump(candidate, include_attributes=False) == ast.dump(original, include_attributes=False)
compile((here/'attention_fwd_triton2_shifted.py').read_text(), 'attention_fwd_triton2_shifted.py', 'exec')
print('PASS: exactly one fixed-BLOCK_M row-base shift; full helper, precision, masks, guards, output and 3D launch preserved')
