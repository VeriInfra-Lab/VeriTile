"""Check exact fixed-128 convolution source delta without a GPU or Torch import."""
import ast
from pathlib import Path
here = Path(__file__).resolve().parent
original = ast.parse((here/'triton_conv2d_fwd_repaired.py').read_text())
candidate = ast.parse((here/'triton_conv2d_fwd_shifted.py').read_text())
function = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name=='conv2d_forward_shifted_kernel')
assertions = [n for n in function.body if isinstance(n, ast.Expr) and
              ast.unparse(n)=='tl.static_assert(BLOCK_SIZE_BATCH_HEIGHT_WIDTH == 128)']
assert len(assertions)==1
function.body.remove(assertions[0])
function.name = 'conv2d_forward_kernel'
class Restore(ast.NodeTransformer):
    count = 0
    launches = 0
    def visit_FunctionDef(self, node):
        if node.name == 'conv2d_forward_shifted': node.name = 'conv2d_forward'
        return self.generic_visit(node)
    def visit_Name(self, node):
        if node.id == 'conv2d_forward_shifted_kernel':
            node.id = 'conv2d_forward_kernel'
            self.launches += 1
        return node
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.LShift):
            assert ast.unparse(node) == 'batch_height_width_pid << 7'
            self.count += 1
            return ast.BinOp(ast.Name('batch_height_width_pid', ast.Load()), ast.Mult(),
                             ast.Name('BLOCK_SIZE_BATCH_HEIGHT_WIDTH', ast.Load()))
        return node
restore=Restore()
candidate=restore.visit(candidate)
assert restore.count==1 and restore.launches==1
assert ast.dump(candidate, include_attributes=False)==ast.dump(original, include_attributes=False)
compile((here/'triton_conv2d_fwd_shifted.py').read_text(), 'triton_conv2d_fwd_shifted.py', 'exec')
print('PASS: exactly one fixed-128 spatial base shift; full precision flags, loops, masks, guards and 3D launch preserved')
