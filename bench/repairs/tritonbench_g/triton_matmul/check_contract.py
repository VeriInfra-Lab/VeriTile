"""Focused guarded-wrapper and source-binding checks. CPU interpreter, no GPU."""
import ast
from contextlib import nullcontext
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['TRITON_INTERPRET'] = '1'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/veritile-triton-matmul-interpreter')
sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'triton_matmul_repaired.py'


def dump(node):
    return ast.dump(node, include_attributes=False)


def expression(text):
    return dump(ast.parse(text, mode='eval').body)


def check_source():
    tree = ast.parse(SOURCE.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'matmul_kernel_repaired')
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'matmul_repaired')
    dots = [n for n in ast.walk(kernel) if isinstance(n, ast.Call) and dump(n.func) == expression('tl.dot')]
    assert len(dots) == 1 and dump(dots[0]) == expression('tl.dot(a, b, accumulator)')
    assert expression('tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)') in [dump(n) for n in ast.walk(kernel)]
    assert expression('accumulator.to(tl.float16)') in [dump(n) for n in ast.walk(kernel)]
    assert expression('accumulator.to(tl.float8e4nv)') in [dump(n) for n in ast.walk(kernel)]
    assert not any(isinstance(n, ast.Attribute) and n.attr in {'multiple_of', 'max_contiguous'} for n in ast.walk(kernel))
    expected = [
        'not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)',
        'a.layout != torch.strided or b.layout != torch.strided',
        'a.dtype not in (torch.float16, torch.float8_e4m3fn) or a.dtype != b.dtype',
        'not a.is_cuda or not b.is_cuda or a.device != b.device',
        'a.ndim != 2 or b.ndim != 2', 'K != K_b',
        'not a.is_contiguous() or not b.is_contiguous()', 'a.is_neg() or b.is_neg()',
        'M >= 2**63 or N >= 2**63 or K >= 2**63', 'M != 0 and N != 0',
    ]
    assert [dump(n.test) for n in wrapper.body if isinstance(n, ast.If)] == [expression(t) for t in expected]
    tests = [dump(n.test) for n in ast.walk(wrapper) if isinstance(n, ast.If)]
    for test in ['max(width * M * K, width * K * N, width * M * N) >= 2**63',
                 'triton.cdiv(M, 128) * triton.cdiv(N, 256) >= 2**31',
                 'start % width != 0 or start + width * t.numel() > 2**64']:
        assert expression(test) in tests
    assert not any(isinstance(n, ast.Attribute) and n.attr in {'clone', 'contiguous', 'resolve_neg', 'reshape', 'flatten'}
                   for n in ast.walk(wrapper))
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text().split('namespace')[0]


class Facade:
    def __init__(self, data, **kwargs):
        self.data, self.device, self.is_cuda = data, torch.device('cuda:0'), True
        self.__dict__.update(kwargs)

    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    @property
    def shape(self): return getattr(self, 'shape_override', self.data.shape)
    @property
    def ndim(self): return len(self.shape)
    def numel(self): return math.prod(self.shape)
    def element_size(self): return self.data.element_size()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return getattr(self, 'neg_override', self.data.is_neg())
    def data_ptr(self):
        assert self.numel() > 0, 'empty input pointer queried'
        return getattr(self, 'pointer_override', self.data.data_ptr())


class Launcher:
    def __init__(self, kernel): self.kernel, self.calls = kernel, 0
    def __getitem__(self, grid):
        def run(a, b, c, m, n, k, *strides, **kwargs):
            self.calls += 1
            assert strides == (k, 1, n, 1, max(n, 1), 1)
            assert kwargs == dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256,
                                  BLOCK_SIZE_K=64 if a.dtype == torch.float16 else 128,
                                  GROUP_SIZE_M=8, num_stages=3 if a.dtype == torch.float16 else 4, num_warps=8)
            return self.kernel[grid](a.data, b.data, c.data, m, n, k, *strides, **kwargs)
        return run


def run():
    check_source()
    spec = importlib.util.spec_from_file_location('triton_matmul_contract', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    launcher = Launcher(module.matmul_kernel_repaired)
    module.matmul_kernel_repaired = launcher
    allocations = []
    def allocate(shape, device, dtype):
        backing = torch.full((math.prod(shape) + 8,), -7, dtype=torch.float32).to(dtype)
        result = Facade(backing[3:3 + math.prod(shape)].reshape(shape), device=device)
        allocations.append((result, backing))
        return result
    module.torch = SimpleNamespace(Tensor=Facade, strided=torch.strided, float16=torch.float16,
        float8_e4m3fn=torch.float8_e4m3fn, cuda=SimpleNamespace(device=lambda _: nullcontext()), empty=allocate)
    a, b = Facade(torch.ones(3, 4, dtype=torch.float16)), Facade(torch.ones(4, 5, dtype=torch.float16))
    rejected = 0
    def reject(left, right):
        nonlocal rejected
        counts = (launcher.calls, len(allocations))
        try:
            module.matmul_repaired(left, right)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('unsupported input admitted')
        assert counts == (launcher.calls, len(allocations))
        rejected += 1
    for index in [0, 1]:
        target = [a, b][index]
        for bad in [None, Facade(target.data.to_sparse()), Facade(target.data.float()),
                    Facade(target.data, is_cuda=False), Facade(target.data, device=torch.device('cuda:1')),
                    Facade(target.data.reshape(-1)), Facade(target.data.T), Facade(target.data, neg_override=True),
                    Facade(target.data.T.contiguous().T), Facade(target.data, pointer_override=2**64 - 2),
                    Facade(target.data, pointer_override=1)]:
            args = [a, b]
            args[index] = bad
            reject(*args)
    reject(a, Facade(torch.ones(4, 5).to(torch.float8_e4m3fn)))
    reject(Facade(a.data, shape_override=(2**63, 4)), b)
    reject(Facade(a.data, shape_override=(1, 2**62)), Facade(b.data, shape_override=(2**62, 1)))
    reject(Facade(a.data, shape_override=(128*2**31, 0)), Facade(b.data, shape_override=(0, 1)))
    # Test exact envelope boundaries without allocating their synthetic shapes.
    class ReachedAllocation(Exception): pass
    def reached(*args, **kwargs): raise ReachedAllocation()
    module.torch.empty = reached
    for left, right in [
        (Facade(a.data, shape_override=(1, 2**62-1), pointer_override=0),
         Facade(b.data, shape_override=(2**62-1, 1), pointer_override=2**63)),
        (Facade(a.data, shape_override=(128*(2**31-1), 0)), Facade(b.data, shape_override=(0, 1))),
        (Facade(a.data, shape_override=(0, 2**63-1)), Facade(b.data, shape_override=(2**63-1, 1))),
    ]:
        try: module.matmul_repaired(left, right)
        except ReachedAllocation: pass
        else: raise AssertionError('boundary input did not reach allocation')
    module.torch.empty = allocate
    torch.manual_seed(123)
    trials = 0
    for dtype in [torch.float16, torch.float8_e4m3fn]:
        cases = []
        for m, n, k in [(1, 1, 0), (3, 5, 7), (129, 257, 65), (1025, 1, 1), (0, 5, 7), (3, 0, 7)]:
            left = torch.zeros(m, k)
            right = torch.zeros(k, n)
            # Exactly representable sums isolate addressing from unvalidated
            # interpreter/native FP8 conversion rounding.
            for lane in sorted({0, k // 2, k - 1}):
                if 0 <= lane < k:
                    left[:, lane] = torch.randint(-1, 2, (m,)).float()
                    right[lane, :] = torch.randint(-1, 2, (n,)).float()
            cases.append((left.to(dtype), right.to(dtype)))
        cases.append(((torch.arange(20) % 3 - 1).reshape(5, 4).to(dtype)[1:4], torch.ones(4, 5).to(dtype)))
        cases.append((torch.as_strided((torch.arange(4) % 3 - 1).to(dtype), (1, 4), (2**40, 1)), torch.ones(4, 5).to(dtype)))
        shared = (torch.arange(16) % 3 - 1).reshape(4, 4).to(dtype)
        cases.append((shared, shared))
        for left, right in cases:
            before_a, before_b = left.view(torch.uint8).clone(), right.view(torch.uint8).clone()
            out = module.matmul_repaired(Facade(left), Facade(right))
            assert out is allocations[-1][0]
            expected = (left.float() @ right.float()).to(dtype)
            assert torch.equal(out.data.view(torch.uint8), expected.view(torch.uint8)), (dtype, left.shape, right.shape, out.data.float(), expected.float())
            assert torch.equal(left.view(torch.uint8), before_a) and torch.equal(right.view(torch.uint8), before_b)
            backing = allocations[-1][1].float()
            assert torch.all(backing[:3] == -7) and torch.all(backing[3 + out.numel():] == -7)
            trials += 1
    print(f'triton_matmul: source/guard binding passed; {rejected} preallocation rejections; '
          f'{trials} CPU-interpreted calls; byte/index envelope boundaries passed. No GPU execution.')


if __name__ == '__main__':
    run()
