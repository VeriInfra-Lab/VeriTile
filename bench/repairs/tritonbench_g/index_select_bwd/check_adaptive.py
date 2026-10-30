"""Replay the actual original and tail-aware wrapper, with their shared JIT."""
import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
# This existing harness runs the original's independent math/guard/frame checks
# and exposes the actual Triton CPU interpreter with mocked CUDA metadata.
import check_contract as checked

spec = importlib.util.spec_from_file_location('bwd_adaptive', HERE / 'index_select_bwd_adaptive.py')
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)
adaptive.torch = checked.repair.torch
adaptive.index_select_cat_bwd_repaired_kernel = checked.repair.index_select_cat_bwd_repaired_kernel

# The complete public wrapper, including all validation, indices and slices,
# differs only in its function name and the BLOCK argument.
old = next(n for n in ast.parse((HERE / 'index_select_bwd_repaired.py').read_text()).body
           if isinstance(n, ast.FunctionDef) and n.name == 'index_select_cat_bwd_repaired')
new = next(n for n in ast.parse((HERE / 'index_select_bwd_adaptive.py').read_text()).body
           if isinstance(n, ast.FunctionDef) and n.name == 'index_select_cat_bwd_adaptive')
new = copy.deepcopy(new)
new.name = old.name
changes = 0
for node in ast.walk(new):
    if isinstance(node, ast.keyword) and node.arg == 'BLOCK':
        assert ast.unparse(node.value) == 'tail_block(size)'
        node.value = ast.Constant(512)
        changes += 1
assert changes == 1
assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
for size in range(513):
    width = adaptive.tail_block(size)
    assert width in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert size <= width <= 512
    assert width == 1 or width // 2 < size

# Reuse the complete guard/frame/interpreter checks on the real candidate.
original_wrapper = checked.repair.index_select_cat_bwd_repaired
checked.repair.index_select_cat_bwd_repaired = adaptive.index_select_cat_bwd_adaptive
for cols in (0, 1, 2, 3, 7, 31, 127, 255, 256, 257, 511, 512, 513, 1025):
    checked.case([], cols)
    checked.case([2, 0, 2, 1, 0], cols, singleton=True)
args = checked.case([0, 2], 3)
checked.reject(args, 1, checked.Tensor(checked.torch.tensor([-1, 2])))
checked.reject(args, 2, checked.Tensor(args[0].value[:2]))
# Every admitted FP32 word is retained by the proof. Include exceptional and
# subnormal arithmetic here without treating this finite check as its proof.
torch = checked.torch
words = torch.tensor([-2147483648, 2143289345, 1, 2139095040, -8388608, 0], dtype=torch.int32)
left = words.view(torch.float32).repeat(3, 1).clone()
right = left.clone()
incoming = torch.tensor([[0., 1., 1., -float('inf'), 3., -0.],
                         [1., 2., -1., 0., 4., 1.]], dtype=torch.float32)
index = checked.Tensor(torch.tensor([1, 1], dtype=torch.int64))
original_wrapper(checked.Tensor(left), index, checked.Tensor(incoming))
adaptive.index_select_cat_bwd_adaptive(checked.Tensor(right), index, checked.Tensor(incoming))
assert torch.equal(left.view(torch.int32), right.view(torch.int32))

print(json.dumps({'complete_wrapper_AST_preserved_except_BLOCK': True,
                  'tail_block_sizes_checked': 513,
                  'actual_candidate_wrapper_cases': 29,
                  'special_word_pair': True,
                  'candidate_kernel': 'shared index_select_cat_bwd_repaired_kernel',
                  'no_numerical_or_GPU_experiment': True}, sort_keys=True))
