"""Source correspondence, scalar guards, host order and CPU-interpreted launches.
CUDA allocation is replaced with ordinary CPU allocation; no GPU claim is made.
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['TRITON_INTERPRET'] = '1'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/veritile-batched-vecmat-interpreter')
sys.dont_write_bytecode = True
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / 'tritonbench_g/batched_vecmat_mult/batched_vecmat_mult.py'
SOURCE = HERE / 'batched_vecmat_mult_int64.py'

def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)

def dump(n): return ast.dump(n, include_attributes=False)
def statements(s): return ast.parse(s).body

old = function(ORIGINAL, 'batched_vecmat_kernel')
new = function(SOURCE, 'batched_vecmat_kernel')
assert list(map(dump, new.body[:4])) == list(map(dump, statements('''
dim_n = tl.cast(dim_n, tl.int64)
dim_k = tl.cast(dim_k, tl.int64)
m_index = tl.program_id(0).to(tl.int64)
n_index = tl.program_id(1).to(tl.int64)
''')))
new.body[:4] = copy.deepcopy(old.body[:2])
assert dump(new) == dump(old), 'unreviewed arithmetic, memory, mask, or launch change'
old = function(ORIGINAL, 'batched_vecmat')
new = function(SOURCE, 'batched_vecmat')
assert dump(new.body[0]) == dump(statements('''
for value in (M, N, K, block_m, block_n, block_k, num_warps, num_stages):
    if type(value) is not int or value < 0:
        raise ValueError("dimensions, blocks and tuning arguments must be nonnegative integers")
''')[0])
new.body = new.body[1:]
assert dump(new) == dump(old), 'allocation/guard/launch order changed'
for name in ['Correct', 'FPEquiv']:
    content = (HERE / f'{name}.lean').read_text()
    assert f'batched_vecmat_mult.{"FPEquiv" if name == "Correct" else "Correct"}' not in content
assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text().split('namespace')[0]

spec = importlib.util.spec_from_file_location('batched_vecmat_candidate', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
allocations = []
backing_allocations = []
fail_at = None

def allocate(kind, shape, device, dtype):
    assert device == 'cuda' and dtype == torch.float32
    if len(allocations) == fail_at:
        raise RuntimeError(f'allocation {kind}')
    # Small integers make the exact Real and FP32 reference coincide here.
    count = int(np.prod(shape))
    margin = 7
    backing = torch.full((count + 2*margin,), 12345.0, dtype=dtype)
    out = backing[margin:margin+count].reshape(shape)
    out.copy_(torch.randint(-2, 3, shape).to(dtype) if kind == 'randn'
              else torch.zeros(shape, dtype=dtype))
    backing_allocations.append((out, backing, backing.clone(), margin, count))
    allocations.append((kind, shape, out))
    return out

module.torch = SimpleNamespace(float32=torch.float32,
    randn=lambda *shape, **kw: allocate('randn', shape, **kw),
    zeros=lambda *shape, **kw: allocate('zeros', shape, **kw))

rejected = 0
class SubInt(int): pass
for index in range(8):
    for bad in [-1, True, 2.0, None, SubInt(2)]:
        args = [4, 4, 8, 2, 2, 4, 4, 1]
        args[index] = bad
        allocations.clear()
        try: module.batched_vecmat(*args)
        except ValueError: pass
        else: raise AssertionError(('admitted unsupported scalar', index, bad))
        assert not allocations
        rejected += 1

for args, error in [
    ([4,4,8,2,2,0], ZeroDivisionError),
    ([4,4,7,0,0,4], AssertionError),
    ([4,4,8,0,0,4], ZeroDivisionError),
    ([3,4,8,2,0,4], AssertionError),
    ([4,4,8,2,0,4], ZeroDivisionError),
    ([4,3,8,2,2,4], AssertionError),
]:
    allocations.clear()
    try: module.batched_vecmat(*args)
    except error: pass
    else: raise AssertionError(('missing ordered guard', args))
    assert [x[0] for x in allocations] == ['randn','randn','zeros']

for index in range(3):
    allocations.clear()
    fail_at = index
    try: module.batched_vecmat(4, 4, 7, 0, 0, 0)
    except RuntimeError: pass
    else: raise AssertionError('allocation failure was swallowed')
    assert len(allocations) == index
fail_at = None

executed = 0
for blocks in [(1,1,1),(2,2,4),(4,8,8),(16,32,64)]:
    bm, bn, bk = blocks
    for scales in [(1,1,1),(2,2,2),(2,1,3),(0,1,1),(1,0,1),(1,1,0),(0,0,0)]:
        m,n,k = (x*y for x,y in zip(blocks, scales))
        allocations.clear()
        backing_allocations.clear()
        out = module.batched_vecmat(m,n,k,bm,bn,bk)
        a,b = allocations[0][2],allocations[1][2]
        expected = (a[:,None,:]*b).sum(-1)
        assert out is allocations[2][2]
        assert out.shape == (m,n) and out.dtype == torch.float32
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert len(backing_allocations) == 3
        for _, backing, snapshot, margin, count in backing_allocations[:2]:
            assert torch.equal(backing, snapshot), 'input backing storage changed'
        _, backing, snapshot, margin, count = backing_allocations[2]
        assert torch.equal(backing[:margin], snapshot[:margin]), 'output prefix changed'
        assert torch.equal(backing[margin+count:], snapshot[margin+count:]), 'output suffix changed'
        executed += 1

# Concrete original bug in an otherwise divisible contiguous allocation.
m,n,k,bm,bn,bk = 32784,512,128,16,32,64
assert m % bm == n % bn == k % bk == 0
pm,pn,c,i,j,e = 2048,0,0,0,0,0
logical = (pm*bm+i)*n*k + (pn*bn+j)*k + c*bk+e
assert logical == 2**31 and logical < m*n*k
original = np.array(logical, dtype=np.uint32).view(np.int32).item()
repaired = np.int64(pm)*bm*n*k + np.int64(pn)*bn*k + np.int64(c)*bk
assert original == -(2**31) and repaired == logical
print(f'{rejected} pre-allocation rejections; ordered allocation/guard failures; '
      f'{executed} Triton CPU-interpreted launches with nonzero-offset Tensor windows, '
      'complete input backing-storage and output-sentinel frames; '
      'int32 overflow witness passed. No GPU execution.')
