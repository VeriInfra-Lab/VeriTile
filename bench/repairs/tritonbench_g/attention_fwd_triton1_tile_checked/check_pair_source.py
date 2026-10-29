"""Check the exact optimization delta without importing Torch or running a GPU."""
import ast
import copy
from pathlib import Path

source = Path(__file__).with_name('attention_fwd_triton1_tile_checked.py')
module = ast.parse(source.read_text())
def function(name):
    return copy.deepcopy(next(n for n in module.body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
def tree(node):
    return ast.dump(node, include_attributes=False)

original = function('attention_fwd_kernel')
candidate = function('attention_fwd_shifted_kernel')
candidate.name = original.name
assert ast.unparse(candidate.body.pop(0)) == 'tl.static_assert(BT == 32)'

class RestoreOffsets(ast.NodeTransformer):
    count = 0
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.LShift):
            assert ast.unparse(node) == 'i << 5'
            self.count += 1
            return ast.BinOp(left=ast.Name(id='i', ctx=ast.Load()), op=ast.Mult(),
                             right=ast.Name(id='BT', ctx=ast.Load()))
        return node

restore = RestoreOffsets()
candidate = restore.visit(candidate)
assert restore.count == 4
assert tree(candidate) == tree(original)

host_original = function('_forward_with_state')
host_candidate = function('_forward_shifted_with_state')
host_candidate.name = host_original.name
class RestoreLaunch(ast.NodeTransformer):
    count = 0
    def visit_Name(self, node):
        if node.id == 'attention_fwd_shifted_kernel':
            self.count += 1
            node.id = 'attention_fwd_kernel'
        return node
launch = RestoreLaunch()
host_candidate = launch.visit(host_candidate)
assert launch.count == 1
assert tree(host_candidate) == tree(host_original)
compile(source.read_text(), str(source), 'exec')
print('PASS: exactly four BT=32 temporal shifts; complete kernel and checked launch preserved')

# The retained original remains the previously accepted repair: a single
# empty-aware static-tile guard, plus the two separately checked new definitions.
previous = source.parent.parent / 'attention_fwd_triton1' / 'Whole001' / 'attention_fwd_triton1_repaired.py'
before = ast.parse(previous.read_text())
after = copy.deepcopy(module)
after.body = [n for n in after.body if not (isinstance(n, ast.FunctionDef) and n.name in
              {'attention_fwd_shifted_kernel', '_forward_shifted_with_state'})]
forward = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_with_state')
guards = [n for n in forward.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'dim * dim > 1048576']
assert len(guards) == 1
forward.body.remove(guards[0])
assert tree(before) == tree(after)
print('PASS: retained original differs from shared source only by accepted static-tile guard')
