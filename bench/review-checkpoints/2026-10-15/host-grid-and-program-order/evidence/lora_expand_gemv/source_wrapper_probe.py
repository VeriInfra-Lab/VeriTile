"""Execute the original wrapper with real CPU tensor metadata and a launch recorder.

This checks wrapper control flow, not compiled GPU or native memory semantics.
The original AST is loaded without executing the file's CUDA test harness.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import traceback

import torch
import triton

REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
SOURCE = REPO / 'bench/tritonbench_g/lora_expand_gemv/lora_expand_gemv.py'
REPORT = Path(__file__).with_name('source-wrapper-probe.json')


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((args, kwargs, grid(kwargs)))
        return launch


source = SOURCE.read_text()
function = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_bgmv_expand')
recorder = Recorder()
environment = {'torch': torch, 'triton': triton, '_bgmv_expand_kernel': recorder}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), environment)
wrapper = environment['_bgmv_expand']

accepted = []
for input_type, weight_type, output_type, index_type, fourth_axis, add, k in itertools.product(
        [torch.float16, torch.bfloat16, torch.float32],
        [torch.float16, torch.bfloat16],
        [torch.float16, torch.bfloat16, torch.float32, torch.int32],
        [torch.int32, torch.int64, torch.float32], [False, True], [False, True], [5, 8]):
    inputs = torch.empty((2, k), dtype=input_type)
    weight_shape = (2, 1, 3, k) if fourth_axis else (2, 3, k)
    weights = torch.empty(weight_shape, dtype=weight_type)
    output = torch.empty((2, 3), dtype=output_type)
    indices = torch.empty((2,), dtype=index_type)
    previous = len(recorder.calls)
    assert wrapper(inputs, weights, output, indices, add) is None
    assert len(recorder.calls) == previous + 1
    args, kwargs, grid = recorder.calls[-1]
    view = weights.squeeze(1) if fourth_axis else weights
    assert args[0] is inputs and args[2] is output and args[5] is indices
    assert args[1].data_ptr() == weights.data_ptr()
    assert tuple(args[1].shape) == tuple(view.shape)
    assert args[3:5] == (3, k)
    assert args[6:] == inputs.stride() + view.stride() + output.stride()
    assert kwargs == {'BLOCK_K': triton.next_power_of_2(k), 'BLOCK_N': 256,
                      'SPLIT_N': 3, 'EVEN_K': k % triton.next_power_of_2(k) == 0,
                      'ADD_INPUTS': add, 'CAST_TYPE': input_type == torch.float32}
    assert grid == (3, 2)
    accepted.append({'input': str(input_type), 'weight': str(weight_type),
                     'output': str(output_type), 'index': str(index_type),
                     'rank': 4 if fourth_axis else 3, 'add_inputs': add, 'K': k})


def regular():
    return [torch.empty((2, 5)), torch.empty((2, 3, 5), dtype=torch.float16),
            torch.empty((2, 3)), torch.empty((2,), dtype=torch.int32)]


errors = []


def expect_error(name, arguments, error_type, source_fragment):
    before = len(recorder.calls)
    try:
        wrapper(*arguments)
    except error_type as error:
        frames = traceback.extract_tb(error.__traceback__)
        frame = next(f for f in reversed(frames) if f.filename == str(SOURCE))
        line = source.splitlines()[frame.lineno - 1].strip()
        assert source_fragment in line, (name, line)
        assert len(recorder.calls) == before
        errors.append({'case': name, 'exception': type(error).__name__,
                       'line': frame.lineno, 'source': line, 'launches': 0})
    else:
        raise AssertionError(f'{name}: expected {error_type.__name__}')


a = regular()
a[0] = torch.empty((), dtype=torch.int32)
a[1] = torch.empty((), dtype=torch.float32)
expect_error('input dtype before all later errors', a, AssertionError, 'assert inputs.dtype')
a = regular()
a[0] = torch.empty(())
a[1] = torch.empty((), dtype=torch.float32)
expect_error('weight dtype before size access', a, AssertionError, 'assert lora_b_weights.dtype')
a = regular()
a[0] = torch.empty(())
expect_error('input size access', a, IndexError, 'inputs.size(1)')
a = regular()
a[1] = torch.empty((), dtype=torch.float16)
expect_error('weight last size access', a, IndexError, 'lora_b_weights.size(-1)')
a = regular()
a[0] = torch.empty((2, 7))
expect_error('inner-size equality', a, AssertionError, 'inputs.size(1)')
a = regular()
a[0] = torch.empty((5, 2)).transpose(0, 1)
expect_error('input contiguous', a, AssertionError, 'assert inputs.is_contiguous()')
a = regular()
a[2] = torch.empty((3, 2)).transpose(0, 1)
expect_error('output contiguous', a, AssertionError, 'assert output_tensor.is_contiguous()')
a = regular()
a[1] = torch.empty((2, 2, 3, 5), dtype=torch.float16)
expect_error('4D singleton', a, AssertionError, 'assert lora_b_weights.size(1) == 1')
a = regular()
a[1] = torch.empty((3, 5), dtype=torch.float16)
expect_error('non-4D rank', a, AssertionError, 'assert lora_b_weights.ndim == 3')
a = regular()
a[1] = torch.empty((2, 5, 3), dtype=torch.float16).transpose(1, 2)
expect_error('weight contiguous', a, AssertionError, 'assert lora_b_weights.is_contiguous()')
for fourth_axis in [False, True]:
    a = regular()
    a[0] = torch.empty((2, 0))
    a[1] = torch.empty((2, 1, 3, 0) if fourth_axis else (2, 3, 0), dtype=torch.float16)
    a[2] = torch.empty(())
    a[3] = torch.empty((), dtype=torch.int32)
    expect_error(f'zero K before index size / output stride, rank {4 if fourth_axis else 3}',
                 a, ZeroDivisionError, 'K % BLOCK_K')
a = regular()
a[3] = torch.empty((), dtype=torch.int32)
expect_error('index batch size', a, IndexError, 'lora_indices_tensor.size(0)')
a = regular()
a[2] = torch.empty((3,))
expect_error('output stride 1', a, IndexError, 'output_tensor.stride(1)')

a = regular()
a[1] = torch.empty((2, 0, 5), dtype=torch.float16)
a[2] = torch.empty((2, 0))
wrapper(*a)
_, zero_n_kwargs, zero_n_grid = recorder.calls[-1]
assert zero_n_kwargs['SPLIT_N'] == 0 and zero_n_grid == (0, 2)

a = regular()
wrapper(*a)
assert recorder.calls[-1][1]['ADD_INPUTS'] is True

helper = Path(triton.__file__)
report = {
    'status': 'passed',
    'source': str(SOURCE),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'torch_version': torch.__version__, 'triton_version': triton.__version__,
    'triton_init': str(helper), 'triton_init_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
    'accepted_combinations': len(accepted),
    'domain': {'inputs': ['float16', 'bfloat16', 'float32'], 'weights': ['float16', 'bfloat16'],
               'outputs': ['float16', 'bfloat16', 'float32', 'int32'],
               'indices': ['int32', 'int64', 'float32'], 'ranks': [3, 4],
               'add_inputs': [False, True], 'K': [5, 8]},
    'error_checks': errors,
    'zero_n': {'SPLIT_N': 0, 'grid': list(zero_n_grid), 'native_launch_validated': False},
    'default_add_inputs': True,
    'zero_k_source_bug': 'next_power_of_2(0)=0; modulo raises before indices/stride/launch',
    'scope': 'Original wrapper executed on real CPU metadata; the GPU kernel is a recorder. '
             'This does not verify native compiled execution, address validity, allocations, '
             'output/index dtype support, or device concurrency.'
}
REPORT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'accepted_combinations': len(accepted),
                  'error_checks': len(errors), 'report': str(REPORT)}))
