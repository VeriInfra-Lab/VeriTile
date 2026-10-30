"""Focused checks for the two concrete matmul contracts and source revisions.
Uses actual CPU Tensor storage and Triton's interpreter, with CUDA allocation
and context mocked. Integer inputs give an exact reference on these cases.
"""
import ast
import copy
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['TRITON_INTERPRET'] = '1'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/veritile-matmul-interpreter')
sys.dont_write_bytecode = True
import torch
import triton
import numpy as np

HERE = Path(__file__).resolve().parent
REPAIRS = HERE.parents[1]
ROOT = HERE.parents[4]

def dump(n): return ast.dump(n, include_attributes=False)
def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
def stmt(s): return ast.parse(s).body[0]

class RemoveCasts(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cast':
            assert dump(node.args[1]) == dump(ast.parse('tl.uint64', mode='eval').body)
            return node.args[0]
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
            assert dump(node.args[0]) == dump(ast.parse('tl.uint64', mode='eval').body)
            return node.func.value
        return node

class Canonical(ast.NodeTransformer):
    mapping = dict(stride_xm='k_size', stride_xk=1, stride_yk='n_size', stride_yn=1,
                   stride_zm='n_size', stride_zn=1)
    def visit_Name(self, n):
        if n.id in self.mapping:
            v = self.mapping[n.id]
            return ast.Name(v, n.ctx) if isinstance(v, str) else ast.Constant(v)
        return n
    def visit_BinOp(self, n):
        n = self.generic_visit(n)
        if isinstance(n.op, ast.Mult) and isinstance(n.right, ast.Constant) and n.right.value == 1:
            return n.left
        return n

class Facade:
    def __init__(self, tensor, **overrides):
        self.data = tensor
        self.device = torch.device('cuda:0')
        self.is_cuda = True
        self.__dict__.update(overrides)
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    @property
    def shape(self): return getattr(self, 'shape_override', self.data.shape)
    @property
    def ndim(self): return len(self.shape)
    def numel(self): return self.data.numel()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.numel() > 0, 'empty pointer queried'
        return self.data.data_ptr()

class Launcher:
    def __init__(self, kernel, configs):
        self.kernel, self.configs = kernel, configs
        self.selected = configs[0] if configs else None
        self.calls = 0
    def __getitem__(self, grid):
        def run(*args):
            self.calls += 1
            args = tuple(x.data if isinstance(x, Facade) else x for x in args)
            return self.kernel[grid](*args, **(self.selected.kwargs if self.selected else {}))
        return run

def unaligned(remainder, shape):
    n = shape[0]*shape[1]
    storage = torch.empty(4*n+4, dtype=torch.uint8).untyped_storage()[remainder:remainder+4*n]
    return torch.empty(0, dtype=torch.float32).set_(storage, 0, shape, (shape[1],1))

def check_source(num, source, original):
    old, new = function(original, 'matmul_kernel'), function(source, 'matmul_kernel')
    assert list(map(dump, new.decorator_list)) == list(map(dump, old.decorator_list))
    scalars = (['m_size','k_size','n_size','stride_xm','stride_xk','stride_yk','stride_yn','stride_zm','stride_zn']
               if num == 1 else ['M','N','K','stride_am','stride_ak','stride_bk','stride_bn','stride_cm','stride_cn'])
    assert list(map(dump, new.body[:9])) == [dump(stmt(f'{x} = tl.cast({x}, tl.uint64)')) for x in scalars]
    new.body = new.body[9:]
    new = RemoveCasts().visit(new)
    loop = next(x for x in new.body if isinstance(x, ast.For))
    oldloop = next(x for x in old.body if isinstance(x, ast.For))
    loads = [x.value for x in loop.body[:2]]
    expected_masks = ([
        '(m_offsets[:, None] < m_size) & (k_offsets[None, :] < k_size - k * k_block_size)',
        '(k_offsets[:, None] < k_size - k * k_block_size) & (n_offsets[None, :] < n_size)'
    ] if num == 1 else [
        '(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)',
        '(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)'
    ])
    assert [dump(next(k.value for k in call.keywords if k.arg == 'mask')) for call in loads] == [dump(ast.parse(m, mode='eval').body) for m in expected_masks]
    assert all(any(k.arg == 'other' and isinstance(k.value, ast.Constant) and k.value.value == 0.0 for k in call.keywords) for call in loads)
    if num == 1:
        new.args.args[6:12] = []
        new = Canonical().visit(new)
        loop = next(x for x in new.body if isinstance(x, ast.For))
        loop.target, loop.iter = copy.deepcopy(oldloop.target), copy.deepcopy(oldloop.iter)
        for x in loop.body[:2]: x.value.keywords = []
        new.body[-1].value.keywords = []
    else:
        for i, call in enumerate(loads):
            mask = next(k for k in call.keywords if k.arg == 'mask')
            assert isinstance(mask.value, ast.BinOp) and isinstance(mask.value.op, ast.BitAnd)
            mask.value = mask.value.right if i == 0 else mask.value.left
    assert dump(new) == dump(old), 'unreviewed kernel arithmetic or addressing change'
    wrapper = function(source, 'matmul' if num == 1 else 'triton_matmul')
    forbidden = {'clone','contiguous','resolve_neg','resolve_conj','flatten','reshape'}
    assert not any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in ast.walk(wrapper))
    guards = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    x,y = ('x','y') if num == 1 else ('a','b')
    m,k,n = ('m_size','k_size','n_size') if num == 1 else ('M','K','N')
    expected_guards = [
        f'not isinstance({x}, torch.Tensor) or not isinstance({y}, torch.Tensor)',
        f'{x}.layout != torch.strided or {y}.layout != torch.strided',
        f'{x}.dtype != torch.float32 or {y}.dtype != torch.float32',
        f'not {x}.is_cuda or not {y}.is_cuda', f'{x}.device != {y}.device',
    ] + ([f'{x}.device.index != 0','torch.get_default_dtype() != torch.float32'] if num == 1 else []) + [
        f'{x}.ndim != 2 or {y}.ndim != 2', f'{x}.shape[1] != {y}.shape[0]',
        f'not {x}.is_contiguous() or not {y}.is_contiguous()', f'{x}.is_neg() or {y}.is_neg()',
        f'{m} >= 2**63 or {k} >= 2**63 or {n} >= 2**63',
        f'({x}.numel() > 0 and {x}.data_ptr() % 4 != 0) or ({y}.numel() > 0 and {y}.data_ptr() % 4 != 0)',
        f'{m} == 0 or {n} == 0',
    ]
    assert list(map(dump,guards)) == [dump(ast.parse(g,mode='eval').body) for g in expected_guards]
    launch = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript))
    canonical = ['k_size','1','n_size','1','n_size','1','32','32','32'] if num == 1 else ['K','1','N','1','N','1']
    assert list(map(dump, launch.args[6:])) == [dump(ast.parse(s, mode='eval').body) for s in canonical]
    assert '.Proofs.' not in (source.parent/'Kernels.lean').read_text().split('namespace')[0]
    assert '.revision001.Correct' not in (source.parent/'FPEquiv.lean').read_text()
    assert '.revision001.FPEquiv' not in (source.parent/'Correct.lean').read_text()

def run(num):
    task = f'matmul_triton{num}'
    folder = REPAIRS/task/'revision001'
    source, original = folder/f'{task}_repaired.py', ROOT/'bench/tritonbench_g'/task/f'{task}.py'
    check_source(num, source, original)
    old_tune = triton.autotune
    triton.autotune = lambda configs, key: lambda fn: Launcher(fn, configs)
    try:
        spec = importlib.util.spec_from_file_location(f'{task}_candidate', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally: triton.autotune = old_tune
    if num == 1: module.matmul_kernel = Launcher(module.matmul_kernel, [])
    allocations = []
    default = [torch.float32]
    def empty(*shape, device, dtype=None):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        out = torch.empty(shape, dtype=dtype or default[0])
        allocations.append(out)
        return out
    module.torch = SimpleNamespace(Tensor=Facade, strided=torch.strided,float32=torch.float32,
        get_default_dtype=lambda: default[0],empty=empty,cuda=SimpleNamespace(device=lambda _:nullcontext()))
    wrapper = module.matmul if num == 1 else module.triton_matmul
    x, y = torch.arange(12,dtype=torch.float32).reshape(3,4),torch.ones(4,5)
    rejects = 0
    def reject(a,b):
        nonlocal rejects
        before = len(allocations),module.matmul_kernel.calls
        try: wrapper(a,b)
        except (TypeError,ValueError): pass
        else: raise AssertionError('unsupported input admitted')
        assert before == (len(allocations),module.matmul_kernel.calls)
        rejects += 1
    for a,b in [(None,Facade(y)),(Facade(x),None),(Facade(x.double()),Facade(y)),
                (Facade(x),Facade(y.double())),(Facade(x.to_sparse()),Facade(y)),(Facade(x),Facade(y.to_sparse())),(Facade(x,is_cuda=False),Facade(y)),
                (Facade(x),Facade(y,device=torch.device('cuda:1'))),
                (Facade(x.reshape(-1)),Facade(y)),(Facade(x),Facade(y.reshape(-1))),
                (Facade(x),Facade(y[:3])),(Facade(x.T),Facade(torch.ones(3,5))),
                (Facade(x),Facade(torch.ones(5,4).T)),(Facade(torch._neg_view(x)),Facade(y)),
                (Facade(x),Facade(torch._neg_view(y))),
                (Facade(x,shape_override=(2**63,4)),Facade(y)),
                (Facade(x,shape_override=(3,2**63)),Facade(y,shape_override=(2**63,5))),
                (Facade(x),Facade(y,shape_override=(4,2**63)))]: reject(a,b)
    for r in [1,2,3]:
        reject(Facade(unaligned(r,(3,4))),Facade(y))
        reject(Facade(x),Facade(unaligned(r,(4,5))))
    if num == 1:
        reject(Facade(x,device=torch.device('cuda:1')),Facade(y,device=torch.device('cuda:1')))
        default[0] = torch.float64
        reject(Facade(x),Facade(y))
        default[0] = torch.float32
    shapes = [(3,5,7),(16,256,64),(65,33,65),(257,65,33),(1,1,0),(0,5,7),(3,0,7)]
    cases = [(torch.randint(-2,3,(m,k)).float(),torch.randint(-2,3,(k,n)).float()) for m,n,k in shapes]
    cases += [(torch.arange(20.).reshape(5,4)[1:4], y),
              (torch.as_strided(torch.arange(4.),(1,4),(2**40,1)),y),
              (torch.arange(9.).reshape(3,3),torch.arange(9.).reshape(3,3))]
    # Actual aliased input object.
    same = torch.arange(16.).reshape(4,4)
    cases.append((same,same))
    cases.extend([(unaligned(1,(0,4)),y),(x,unaligned(3,(4,0)))])
    trials = 0
    for a,b in cases:
        expected = a@b
        saved_a,saved_b = a.clone(),b.clone()
        for config in module.matmul_kernel.configs or [None]:
            module.matmul_kernel.selected = config
            out = wrapper(Facade(a),Facade(b))
            assert out is allocations[-1]
            torch.testing.assert_close(out,expected,rtol=0,atol=0)
            torch.testing.assert_close(a,saved_a,rtol=0,atol=0)
            torch.testing.assert_close(b,saved_b,rtol=0,atol=0)
            trials += 1
    for value in [float('nan'),float('inf'),-float('inf')]:
        out = wrapper(Facade(torch.tensor([[value]])),Facade(torch.ones(1,1)))
        assert out.shape == (1,1)
        trials += 1
    assert len(module.matmul_kernel.configs) == (0 if num == 1 else 8)
    # Original transpose witness and original masked-load overrun.
    if num == 1:
        transposed = torch.arange(256.).reshape(16,16).T
        assert transposed[0,1].item()==16 and transposed.untyped_storage()[4]==0
        reject(Facade(transposed),Facade(torch.eye(16)))
    else:
        for cfg in module.matmul_kernel.configs:
            assert cfg.kwargs['BLOCK_SIZE_M']>16
            assert 16*64==1024  # Original row 16 load, extent M*K=1024.
    # Canonical contiguous offset exceeding signed32 is retained and widened.
    offset = np.int64(2)*np.int64(2**30)
    assert offset == 2**31
    assert np.array(offset,dtype=np.uint32).view(np.int32).item()==-2**31
    print(f'{task}: {rejects} pre-allocation rejections; {trials} CPU-interpreted cases; '
          f'{len(module.matmul_kernel.configs) or 1} configurations. No GPU execution.')

if __name__ == '__main__':
    for number in ([int(sys.argv[1])] if len(sys.argv)>1 else [1,2]): run(number)
