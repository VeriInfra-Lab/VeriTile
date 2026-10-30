"""Actual Triton CPU-interpreter source checks and exact unchanged-wrapper audit."""
import os
os.environ['TRITON_INTERPRET'] = '1'
import ast
import hashlib
import json
import math
from pathlib import Path
import torch
import triton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / 'triton_argmax_nan_first.py'
ORIGINAL = ROOT / 'bench/repairs/tritonbench_g/triton_argmax/triton_argmax_repaired.py'
OUT = ROOT / '.veritile-cache/kernel-evidence/triton_argmax_nan_first'
NAMES = {'argmax_kernel_1_repaired', 'argmax_kernel_2_repaired', 'argmax_kernel_repaired'}
original, candidate = ast.parse(ORIGINAL.read_text()), ast.parse(SOURCE.read_text())
def outer(tree):
    return ast.dump(ast.Module([node for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in NAMES)], []), include_attributes=False)
assert outer(original) == outer(candidate)
scope = {}
nodes = [node for node in candidate.body if isinstance(node, (ast.Import, ast.ImportFrom)) or
         isinstance(node, ast.FunctionDef) and node.name in NAMES]
exec(compile(ast.Module(nodes, []), str(SOURCE), 'exec'), scope)

def first(values):
    nan = [i for i, value in enumerate(values) if isinstance(value, float) and math.isnan(value)]
    return nan[0] if nan else max(range(len(values)), key=lambda i: (values[i], -i))

def staged(values, block, dtype):
    indices = [start + first(values[start:start+block]) for start in range(0, len(values), block)]
    mid = torch.tensor([values[i] for i in indices], dtype=torch.float32)
    if dtype.is_floating_point:
        mid = mid.to(dtype)
    else:
        signed = dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)
        bits = 8 if dtype == torch.bool else 8 * torch.empty((), dtype=dtype).element_size()
        exponent = bits - 1 if signed else bits
        lower = -(1 << exponent) if signed else 0
        upper = (1 << exponent) - (1 << max(exponent-24, 0))
        mid = torch.where(mid == mid, mid, 0).clamp(lower, upper)
        mid = mid.to(torch.int8).ne(0) if dtype == torch.bool else mid.to(dtype)
    return indices[first(mid.tolist())]

records = []
global_cases = [
    ('nan_first', [float('nan'), 1.0], torch.float32),
    ('nan_last', [1.0, float('nan')], torch.float32),
    ('nans_across_blocks', [float('nan'), 1.0, float('nan'), 2.0], torch.float32),
    ('negative_infinity_tie', [-float('inf')] * 17, torch.float32),
    ('positive_infinity_tie', [1.0, float('inf'), -0.0, float('inf')], torch.float32),
    ('signed_zero_tie', [-0.0, 0.0, -0.0], torch.float32),
    ('rounding_retained', [1.0001]+[-9.0]*3+[1.0002]+[-9.0]*11, torch.float16),
    ('integer_nan_conversion', [float('nan'), 1.0, 7.0, 8.0], torch.int32),
    ('bool_byte_conversion', [0.2]+[-9.0]*3+[0.8]+[-9.0]*11, torch.bool),
]
for name, values, dtype in global_cases:
    x = torch.tensor(values, dtype=torch.float32)
    original_bytes = x.view(torch.uint8).clone()
    block = triton.next_power_of_2(math.ceil(math.sqrt(len(values))))
    blocks = triton.cdiv(len(values), block)
    mid_block = triton.next_power_of_2(blocks)
    mid = torch.empty((blocks,), dtype=dtype)
    indices = torch.zeros((mid_block,), dtype=torch.int64)
    output = torch.tensor([-991, -992, -991], dtype=torch.int64)
    scope['argmax_kernel_1_repaired'][(blocks,)](x, mid, indices, len(values), 0, block)
    scope['argmax_kernel_2_repaired'][(1,)](mid, indices, output[1:2], blocks, mid_block)
    expected = staged(values, block, dtype)
    assert output[1].item() == expected, (name, output.tolist(), expected)
    assert torch.equal(x.view(torch.uint8), original_bytes) and output[0] == output[2] == -991
    if dtype == torch.float32:
        assert expected == torch.argmax(x).item()
    records.append({'branch': 'global', 'case': name, 'dtype': str(dtype), 'output': expected,
                    'torch_argmax': torch.argmax(x).item(), 'block': block, 'blocks': blocks,
                    'input_bytes_preserved': True, 'output_guards_preserved': True})

dimension_cases = [('nan_at_one', [1.0, float('nan')]),
                   ('nan_between', [-3.0, float('nan'), 2.0]),
                   ('finite_tie', [-3.0, 2.0, 2.0]),
                   ('signed_zero_tie', [-0.0, 0.0]),
                   ('infinity_tie', [-float('inf'), -float('inf')]),
                   ('late_tile_nan', [1.0]*130+[float('nan')]+[2.0]*129),
                   ('early_and_late_nan', [1.0, float('nan')]+[2.0]*127+[float('nan')]+[3.0]*130)]
for name, values in dimension_cases:
    # Multiple outer/inner rows exercise the full m*N*K+n*K+k layout.
    x = torch.tensor(values, dtype=torch.float32).view(1, len(values), 1).repeat(2, 1, 3)
    before = x.view(torch.uint8).clone()
    output = torch.full((8,), -771, dtype=torch.int64)
    scope['argmax_kernel_repaired'][(3,)](x, output[1:7], 2, len(values), 3, 0,
                                        BLOCK_M=128, BLOCK_N=128, INT64_INDEX=True)
    expected = torch.argmax(x, dim=1).reshape(-1)
    assert torch.equal(output[1:7], expected), (name, output.tolist(), expected.tolist())
    assert output[0] == output[-1] == -771 and torch.equal(x.view(torch.uint8), before)
    records.append({'branch': 'dimension', 'case': name, 'shape': list(x.shape),
                    'output': expected.tolist(), 'input_bytes_preserved': True, 'output_guards_preserved': True})

report = {'status': 'PASS', 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'original_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
          'wrapper_and_guards_unchanged_ast': True, 'triton': triton.__version__, 'torch': torch.__version__,
          'execution': 'actual candidate JIT under Triton CPU interpreter; independent scalar staged target',
          'gpu_executed': False, 'records': records, 'deferred_numerical_experiments': []}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'candidate-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'status': 'PASS', 'cases': len(records), 'wrapper_ast_unchanged': True}))
