"""Whole original/candidate source delta, including the complete unchanged host."""
import ast
import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = ast.parse((HERE / 'int8_dequant_matmul_scalar_cast.py').read_text())
candidate = ast.parse((HERE / 'int8_dequant_matmul_reuse_indices.py').read_text())
left = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == '_int8_matmul_rowwise_dequantize')
right = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == left.name)
assert [n.targets[0].id for n in left.body[25:27]] == ['rm', 'rn']
assert ast.dump(left.body[18]) == ast.dump(left.body[25])
assert ast.dump(left.body[19]) == ast.dump(left.body[26])
assert len(left.body) == 37 and len(right.body) == 35
right.body[25:25] = copy.deepcopy(left.body[25:27])
assert ast.dump(original) == ast.dump(candidate), 'Unexpected source or host modification'

lean = (HERE / 'Kernels.lean').read_text()
left = lean[lean.index('def kernel '):lean.index('/-- Candidate reuses')].strip()
right = lean[lean.index('def reuseIndicesKernel '):lean.index('/-- Compatibility names')].strip()
block = '  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))\n  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))\n'
assert left.count(block) == 2 and right.count(block) == 1
first = left.index(block)
second = left.index(block, first + len(block))
left = left[:second] + left[second+len(block):]
right = right.replace('def reuseIndicesKernel ', 'def kernel ', 1)
assert left == right, 'Unexpected Lean source modification'
print('PASS: complete Python and Lean candidates remove exactly duplicate rm/rn assignments; 35/37 source statements and all original host/branches retained.')
