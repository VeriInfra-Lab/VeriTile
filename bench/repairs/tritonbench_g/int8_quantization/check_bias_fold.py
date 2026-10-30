"""Check the exact source edit and reuse the complete actual-source CPU harness."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = HERE / 'int8_quantization_repaired.py'
candidate = HERE / 'int8_quantization_bias_fold.py'
left, right = ast.parse(original.read_text()), ast.parse(candidate.read_text())
right.body[0] = left.body[0]
kernel = next(n for n in right.body if isinstance(n, ast.FunctionDef) and n.name == '_per_block_int8_repaired')
bias = next(n for n in kernel.body if isinstance(n, ast.Assign) and n.targets[0].id == 'biased')
assert ast.unparse(bias.value) == 'normalized + tl.where(normalized >= 0.0, 0.5, -0.5)'
bias.value = ast.parse('normalized + 0.5 * tl.where(normalized >= 0.0, 1.0, -1.0)', mode='eval').body
assert ast.dump(left) == ast.dump(right), 'Candidate has an unexpected source or host change'

lean = (HERE / 'Kernels.lean').read_text()
left_lean = lean[lean.index('def source (p : Parameters)'):lean.index('/-- Eliminate the per-lane')].strip()
right_lean = lean[lean.index('def biasFoldSource (p : Parameters)'):lean.index('def originalSHA256')].strip()
right_lean = right_lean.replace('def biasFoldSource', 'def source').replace(
    'tl.where(normalized >= 0.0, 0.5, -0.5)',
    '0.5 * tl.where(normalized >= 0.0, 1.0, -1.0)')
assert left_lean == right_lean, 'Lean candidate has an unexpected change'

harness = (HERE / 'check_contract.py').read_text()
assert harness.count("'int8_quantization_repaired.py'") == 1
assert harness.count('dfd74b467912d86e71cbb195af991c362aa246b95639bb062dd52e6c2ca5f6d0') == 1
candidate_kernel = next(n for n in ast.parse(candidate.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_per_block_int8_repaired')
pin = hashlib.sha256(ast.dump(candidate_kernel).encode()).hexdigest()
harness = harness.replace("'int8_quantization_repaired.py'", "'int8_quantization_bias_fold.py'")
harness = harness.replace('dfd74b467912d86e71cbb195af991c362aa246b95639bb062dd52e6c2ca5f6d0', pin)
namespace = {'__file__': str(HERE / 'check_contract.py'), '__name__': '__main__'}
exec(compile(harness, str(HERE / 'check_contract.py'), 'exec'), namespace)
print('bias-fold pair: whole module AST differs only in the expected constant fold; full CPU harness passed')
