"""Source pins and integer/domain audit only; no floating-point experiment."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECTED = HERE / 'rbe_triton_transform_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/rbe_triton_transform/rbe_triton_transform.py'
PINS = {
    SELECTED: '9095bec159a1bceccea0b6415d198eb1efed6fd30947e79d378ff6f70f17c592',
    ORIGINAL: '57ff14376b6780d4c8cca0a590189d279a2e4e1dfffd95c4be529440c1dd57af',
}
for path, expected in PINS.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
source = SELECTED.read_text()
module = ast.parse(source)
functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
kernel = functions['rbe_triton']
assignments = {node.targets[0].id: node.value for node in kernel.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
for name, expression in {
    'out_real': 'real * cos - imag * sin',
    'out_imag': 'real * sin + imag * cos',
    'x_real_mask': '(offs_m[:, None] < M) & (offs_n[None, :] < K)',
    'x_imag_mask': '(offs_m[:, None] < M) & (1 + offs_n[None, :] < K)',
    'start_block': 'start_token_position + pid_m * BLOCK_SIZE_M',
}.items():
    assert ast.dump(assignments[name]) == ast.dump(ast.parse(expression, mode='eval').body), name
for name in ['M', 'K', 'stride_x_batch', 'stride_x_m', 'stride_x_n',
             'stride_out_batch', 'stride_out_m', 'stride_out_n', 'start_token_position']:
    assert ast.unparse(assignments[name]) == f'tl.cast({name}, tl.int64)', name
assert 'x_ptrs + stride_x_n' in source
assert 'out_ptrs + stride_out_n' in source
assert source.count('tl.debug_barrier()') == 2
helper = ast.unparse(functions['get_freq_multi_tokens'])
for text in ['freqs.to(tl.float32) / DIM', 'tl.extra.cuda.libdevice.pow(theta, freqs)',
             'tl.arange(0, NB_TOKENS).to(tl.int64)', 'tl.cast(starting_idx, tl.int64)',
             'tl.cos(freqs)', 'tl.sin(freqs)']:
    assert text in helper, text
# Exact mathematical integer boundary computations, with no GPU launch.
assert ((131071 + 1) // 2) * ((1 + 1023) // 1024) == 65536
assert -(1 << 63) + 1 < (1 << 63)
assert ((1 << 63) - 2) + 2 - 1 < (1 << 63)
assert not ((1 << 63) - 1) + 2 - 1 < (1 << 63)
print(json.dumps({
    'selected_sha256': PINS[SELECTED], 'original_sha256': PINS[ORIGINAL],
    'source_formula_and_masks': 'pass', 'integer_casts_and_stride_neighbors': 'pass',
    'helper_frequency_cast_and_trig': 'pass',
    'supported_launch_counterexample': {'shape': [1, 131071, 1], 'position': 0, 'grid': [1, 65536]},
    'original_stride_counterexample': {'backing': [1, 2, 3, 4], 'shape': [1, 1, 2],
                                      'last_stride': 2, 'logical_pair': [1, 3], 'loaded_pair': [1, 2]},
    'numerical_experiments': 0, 'gpu_runs': 0,
}, indent=2))
