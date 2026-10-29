"""Replay the unmodified wrapper AST with CPU allocations and a launch spy.

This does not execute Triton or validate CUDA allocation/device behavior.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import torch
import triton
from triton.language.semantic import TritonSemantic

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
SOURCE = ROOT / 'bench/tritonbench_g/matrix_transpose/matrix_transpose.py'
OUT = Path('/tmp/veritile-standardization-20261015/matrix-host-wrapper-probe.json')


class TorchCPU:
    float16 = torch.float16

    def __init__(self):
        self.calls = []

    def factory(self, name, shape, **kwargs):
        assert kwargs == {'dtype': torch.float16, 'device': 'cuda'}
        self.calls.append({'factory': name, 'shape': list(shape), 'requested_device': kwargs['device'], 'dtype': str(kwargs['dtype'])})
        return getattr(torch, name)(shape, dtype=kwargs['dtype'], device='cpu')

    def randn(self, shape, **kwargs):
        return self.factory('randn', shape, **kwargs)

    def zeros(self, shape, **kwargs):
        return self.factory('zeros', shape, **kwargs)


class KernelSpy:
    def __init__(self):
        self.calls = []
        self.returned = None

    def __getitem__(self, grid):
        assert grid == (1,)

        def launch(matrix, out, sx, sy, ox, oy, m, d):
            assert (sx, sy) == matrix.stride()
            assert (ox, oy) == out.stride()
            assert matrix.dtype == out.dtype == torch.float16
            assert tuple(matrix.shape) == (m, d)
            assert tuple(out.shape) == (d, m)
            assert matrix.is_contiguous() and out.is_contiguous()
            assert torch.count_nonzero(out).item() == 0
            self.returned = out
            self.calls.append({'grid': list(grid), 'input_shape': list(matrix.shape), 'output_shape': list(out.shape), 'input_strides': [sx, sy], 'output_strides': [ox, oy], 'dtype': str(matrix.dtype), 'initial_output_zero': True, 'input_contiguous': True, 'output_contiguous': True, 'launch_dimensions': [m, d]})

        return launch


tree = ast.parse(SOURCE.read_text())
wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'wrapper')
module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
cases = [(16, 16), (32, 64), (1, 1), (0, 3), (3, 0), (0, 0), (3, 5), (-1, 4), (4, -1), (1.5, 4), (True, 4)]
rows = []
for m, d in cases:
    torch_proxy, spy = TorchCPU(), KernelSpy()
    env = {'torch': torch_proxy, 'kernel': spy}
    exec(compile(module, str(SOURCE), 'exec'), env)
    row = {'dimensions': [m, d]}
    try:
        returned = env['wrapper'](m, d)
    except Exception as exc:
        row.update({'status': 'host_error_before_launch', 'exception_type': type(exc).__name__, 'exception': str(exc), 'allocation_calls': torch_proxy.calls, 'launch_count': len(spy.calls)})
        assert not spy.calls
        assert (isinstance(m, int) and not isinstance(m, bool) and m < 0) or (isinstance(d, int) and d < 0) or isinstance(m, float) or isinstance(m, bool)
    else:
        assert len(torch_proxy.calls) == 2 and len(spy.calls) == 1
        assert returned is spy.returned
        assert spy.calls[0]['input_strides'] == [max(d, 1), 1]
        assert spy.calls[0]['output_strides'] == [max(m, 1), 1]
        row.update({'status': 'wrapper_metadata_replayed', 'allocation_calls': torch_proxy.calls, 'launch': spy.calls[0], 'returns_allocated_output_identity': True, 'pre_fix_input_stride_match': spy.calls[0]['input_strides'] == [d, 1], 'pre_fix_output_stride_match': spy.calls[0]['output_strides'] == [m, 1]})
    rows.append(row)

# These invalid domains reject before builder access. This exercises installed
# semantic validation, not an offline/native compilation or a GPU launch.
arange = []
for end, expected in [(0, 'greater than the start'), (3, 'power of 2'), (5, 'power of 2'), (2**32, 'fit in int32')]:
    try:
        TritonSemantic.arange(object(), 0, end)
    except ValueError as exc:
        assert expected in str(exc)
        arange.append({'start': 0, 'end': end, 'status': 'rejected', 'diagnostic': str(exc)})
    else:
        raise AssertionError('invalid arange accepted')

receipt = {
    'status': 'pass',
    'source': str(SOURCE),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'wrapper_ast_sha256': hashlib.sha256(ast.dump(wrapper, include_attributes=False).encode()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'torch_version': torch.__version__,
    'triton_version': triton.__version__,
    'execution': 'original wrapper AST; CUDA allocation requests intercepted and delegated to CPU; kernel replaced with launch spy',
    'gpu_execution': False,
    'native_compilation': False,
    'cases': rows,
    'installed_arange_validation': arange,
    'installed_arange_validator_sha256': hashlib.sha256(inspect.getsource(TritonSemantic.arange).encode()).hexdigest(),
    'limits': ['No CUDA allocator correspondence', 'No execution of the Triton kernel', 'No proof of complete Python caller error domain', 'No RNG or numeric arithmetic claim', 'Positive power-of-two dimensions are necessary for arange; not asserted sufficient for complete native launch'],
}
OUT.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'cases': len(rows), 'invalid_arange_cases': len(arange), 'output': str(OUT)}))
