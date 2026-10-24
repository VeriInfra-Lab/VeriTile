"""Probe the original wrapper AST, observing allocations and launch metadata.

CUDA allocation requests are recorded then serviced by real CPU Torch tensors;
the Triton kernel is replaced by a recorder. No GPU behavior is claimed.
"""
import ast
import hashlib
import json
from pathlib import Path
import traceback

import torch
import triton

REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
SOURCE = REPO / 'bench/tritonbench_g/batched_vecmat_mult/batched_vecmat_mult.py'
REPORT = Path(__file__).with_name('source-wrapper-probe.json')
source = SOURCE.read_text()
function = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'batched_vecmat')


class TorchSpy:
    float32 = torch.float32

    def __init__(self):
        self.events = []
        self.fail_at = None

    def allocate(self, init, shape, kwargs):
        self.events.append({'operation': init, 'shape': list(shape),
                            'device': kwargs['device'], 'dtype': str(kwargs['dtype'])})
        assert kwargs == {'device': 'cuda', 'dtype': torch.float32}
        if self.fail_at == len(self.events):
            raise RuntimeError('injected allocation failure')
        return getattr(torch, init)(*shape, device='cpu', dtype=torch.float32)

    def randn(self, *shape, **kwargs):
        return self.allocate('randn', shape, kwargs)

    def zeros(self, *shape, **kwargs):
        return self.allocate('zeros', shape, kwargs)


class KernelSpy:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


allocator = TorchSpy()
kernel = KernelSpy()
environment = {'torch': allocator, 'batched_vecmat_kernel': kernel}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), environment)
wrapper = environment['batched_vecmat']

accepted = []
cases = [
    (4, 6, 4, 2, 3, 2, 4, 1),
    (4, 6, 4, 2, 3, 2, 2, 2),
    (0, 6, 4, 2, 3, 2, 4, 1),
    (4, 0, 4, 2, 3, 2, 4, 1),
    (4, 6, 0, 2, 3, 2, 4, 1),
    (0, 0, 0, 2, 3, 2, 4, 1),
    (6, 10, 6, 3, 5, 3, 4, 1),
    (4, 6, 4, -2, 3, 2, 4, 1),
    (4, 6, 4, 2, -3, 2, 4, 1),
    (4, 6, 4, 2, 3, -2, 4, 1),
    (4, 6, 4, 2, 3, 2, 0, 0),
    (4, 6, 4, 2, 3, 2, -1, -1),
]
for case in cases:
    M, N, K, BM, BN, BK, warps, stages = case
    allocator.events = []
    previous = len(kernel.calls)
    output = wrapper(*case)
    assert len(kernel.calls) == previous + 1
    grid, args, kwargs = kernel.calls[-1]
    assert grid == (M // BM, N // BN)
    assert args[2:5] == (M, N, K)
    assert args[5] is output
    assert tuple(args[0].shape) == (M, K) and tuple(args[1].shape) == (M, N, K)
    assert tuple(output.shape) == (M, N)
    assert all(t.dtype == torch.float32 and t.is_contiguous() for t in [args[0], args[1], output])
    assert bool((output == 0).all())
    assert kwargs == {'block_m': BM, 'block_n': BN, 'block_k': BK,
                      'num_warps': warps, 'num_stages': stages}
    assert [e['shape'] for e in allocator.events] == [[M, K], [M, N, K], [M, N]]
    assert [e['operation'] for e in allocator.events] == ['randn', 'randn', 'zeros']
    accepted.append({'arguments': list(case), 'grid': list(grid),
                     'events': allocator.events.copy(),
                     'lean_natural_domain': min(case) >= 0,
                     'native_specialization_admitted': False})

errors = []


def expect_error(name, args, error_type, expected_allocations, fragment=None, fail_at=None):
    allocator.events = []
    allocator.fail_at = fail_at
    before = len(kernel.calls)
    try:
        wrapper(*args)
    except error_type as error:
        frames = traceback.extract_tb(error.__traceback__)
        frame = next(f for f in reversed(frames) if f.filename == str(SOURCE))
        line = source.splitlines()[frame.lineno - 1].strip()
        if fragment is not None:
            assert fragment in line, (name, line)
        assert len(allocator.events) == expected_allocations
        assert len(kernel.calls) == before
        errors.append({'case': name, 'exception': type(error).__name__,
                       'source_line': frame.lineno, 'source': line,
                       'events': allocator.events.copy(), 'launches': 0})
    else:
        raise AssertionError(name + ': expected exception')
    finally:
        allocator.fail_at = None


expect_error('zero BK first', (4, 6, 4, 0, 0, 0), ZeroDivisionError, 3, 'K % block_k')
expect_error('K divisibility before zero BM', (4, 6, 5, 0, 0, 2), AssertionError, 3, 'K % block_k')
expect_error('zero BM second', (4, 6, 4, 0, 0, 2), ZeroDivisionError, 3, 'M % block_m')
expect_error('M divisibility before zero BN', (5, 6, 4, 2, 0, 2), AssertionError, 3, 'M % block_m')
expect_error('zero BN third', (4, 6, 4, 2, 0, 2), ZeroDivisionError, 3, 'N % block_n')
expect_error('N divisibility last', (4, 7, 4, 2, 3, 2), AssertionError, 3, 'N % block_n')
expect_error('negative M allocation before zero BK', (-1, 6, 4, 2, 3, 0), RuntimeError, 1, 'torch.randn(M, K')
expect_error('negative K allocation before zero BK', (4, 6, -1, 2, 3, 0), RuntimeError, 1, 'torch.randn(M, K')
expect_error('negative N second allocation', (4, -1, 4, 2, 3, 0), RuntimeError, 2, 'torch.randn(M, N, K')
for fail_at, fragment in [(1, 'torch.randn(M, K'), (2, 'torch.randn(M, N, K'), (3, 'torch.zeros(M, N')]:
    expect_error(f'allocation failure {fail_at} before zero BK', (4, 6, 4, 2, 3, 0),
                 RuntimeError, fail_at, fragment, fail_at)

allocator.events = []
wrapper(4, 6, 4, 2, 3, 2)
assert kernel.calls[-1][2]['num_warps'] == 4 and kernel.calls[-1][2]['num_stages'] == 1
report = {'status': 'passed', 'source': str(SOURCE),
          'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'torch_version': torch.__version__, 'triton_version': triton.__version__,
          'accepted_cases': accepted, 'error_checks': errors,
          'defaults': {'num_warps': 4, 'num_stages': 1},
          'scope': 'Original wrapper AST with allocation/device redirection and kernel recorder. '
                   'No numerical GPU output, native compilation, RNG distribution, or CUDA allocator proof.',
          'structural_findings': [
              'All three fp32 CUDA allocation requests precede divisibility and zero-block errors.',
              'Natural zero dimensions and zero K reach the launch recorder; native admission remains separate.',
              'Non-power-of-two blocks pass source guards, although native arange requirements remain open.',
              'Negative blocks may pass source guards and produce negative grids; outside the natural Lean metadata domain.',
              'num_warps/num_stages are passed unchanged, including zero/negative values; native legality is not asserted.'
          ]}
REPORT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'accepted_cases': len(accepted),
                  'error_checks': len(errors), 'report': str(REPORT)}))
