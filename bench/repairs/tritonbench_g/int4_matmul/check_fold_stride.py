"""Check the complete source delta, including unchanged host/helpers/configs."""
import ast
import copy
from pathlib import Path
P=Path(__file__).resolve().parent
left=ast.parse((P/'int4_matmul_repaired.py').read_text())
right=ast.parse((P/'int4_matmul_fold_stride.py').read_text())
class Restore(ast.NodeTransformer):
    def __init__(self): self.count=0
    def visit_BinOp(self,node):
        self.generic_visit(node)
        if ast.dump(node)==ast.dump(ast.parse('k * (BLOCK_SIZE_K * SPLIT_K)',mode='eval').body):
            self.count+=1
            return ast.parse('k * BLOCK_SIZE_K * SPLIT_K',mode='eval').body
        return node
restore=Restore(); right=restore.visit(right)
assert restore.count==3
assert ast.dump(left)==ast.dump(right), 'unexpected Python/host/helper change'
s=(P/'Kernels.lean').read_text(); a=s.index('def kernel (p : Parameters)'); b=s.index('/-- Precombine',a)
left=s[a:b].strip(); a=s.index('def foldStrideKernel (p : Parameters)'); b=s.index('\nend\nend VeriTile.Bench.Repairs.Int4Matmul.KernelSource',a)
right=s[a:b].strip().replace('def foldStrideKernel','def kernel',1)
assert right.count('k * $(p.blockInner * p.splits)')==3
right=right.replace('k * $(p.blockInner * p.splits)','k * $(p.blockInner) * $(p.splits)')
assert left==right, 'unexpected complete Lean quotation change'
print('PASS: complete Python and Lean sources change exactly three dynamic k*BK*S expressions to k*(BK*S); host/helpers/configurations unchanged.')
