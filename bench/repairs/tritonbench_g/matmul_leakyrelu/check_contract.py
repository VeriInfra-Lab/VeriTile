"""Targeted checked-wrapper/source tests. CUDA allocation and launch are mocked."""
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch

TASK = "matmul_leakyrelu"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / (TASK + "_repaired.py")
ORIGINAL = HERE.parents[2] / "tritonbench_g" / TASK / (TASK + ".py")
INT8 = TASK == "int8_matmul_kernel"

class Tensor:
    def __init__(self, shape, dtype=torch.float32, *, strides=None, contiguous=True,
                 negative=False, data=None, device=0, cuda=True, layout=torch.strided):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.layout = layout
        self.device = SimpleNamespace(index=device)
        self.is_cuda = cuda
        self._contiguous, self._negative = contiguous, negative
        self._strides = strides if strides is not None else (max(shape[-1], 1), 1)
        self.data = torch.tensor([0.0]) if data is None else data
    def numel(self):
        return self.shape[0]*self.shape[1]
    def data_ptr(self): return getattr(self, "pointer", 4096)
    def stride(self, axis=None):
        return self._strides if axis is None else self._strides[axis]
    def is_contiguous(self): return self._contiguous
    def is_neg(self): return self._negative

class Runtime:
    Tensor = Tensor
    strided = torch.strided
    float32, float16, uint8, int32 = torch.float32, torch.float16, torch.uint8, torch.int32
    isfinite = staticmethod(lambda a: torch.isfinite(a.data))
    trunc = staticmethod(lambda a: torch.trunc(a.data))
    def __init__(self, failure=None):
        self.events, self.failure = [], failure
        self.current_device, self.launch_devices = 0, []
        self.cuda = SimpleNamespace(current_device=lambda: self.current_device, device=self.device)
    @contextmanager
    def device(self, device):
        previous = self.current_device
        self.current_device = device.index
        try: yield
        finally: self.current_device = previous
    def empty(self, shape, device, dtype):
        self.events.append("allocation")
        if self.failure == "allocation": raise RuntimeError("allocation")
        self.output = Tensor(shape, dtype=dtype, device=device.index)
        if self.failure == "outputAddress": self.output.pointer=2**64-4
        assert dtype == (self.int32 if INT8 else self.float16)
        return self.output
    def __getitem__(self, grid):
        self.grid = grid
        return self.launch
    def launch(self, a, b, c, m, n, k, am, ak, bk, bn, cm, cn, **kwargs):
        assert self.current_device == a.device.index == b.device.index == c.device.index
        self.launch_devices.append(self.current_device)
        self.events.append("launch")
        assert c is self.output and c.shape == (m, n)
        assert a.shape == (m, k) and b.shape == ((k // 4 if INT8 else k), n)
        assert (am, ak, bk, bn, cm, cn) == (*a.stride(), *b.stride(), *c.stride())
        if INT8:
            assert kwargs == {}
            for cfg in self.configs:
                meta = cfg.kwargs
                assert self.grid(meta) == (((m + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M']) * ((n + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N']),)
        else:
            assert kwargs == dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
                                  GROUP_SIZE_M=4, ACTIVATION=self.activation)
            assert self.grid(kwargs) == (((m + 31) // 32) * ((n + 31) // 32),)
        if self.failure == "launch": raise RuntimeError("launch")


def instantiate(failure=None):
    runtime = Runtime(failure)
    def config(kwargs, **options): return SimpleNamespace(kwargs=kwargs, **options)
    ns = {"torch": runtime, "triton": SimpleNamespace(cdiv=lambda x,y:(x+y-1)//y, Config=config), "matmul_kernel": runtime}
    nodes = ast.parse(SOURCE.read_text()).body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in {"matmul", "get_autotune_config"}
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_unchecked_matmul" for t in node.targets)]
    exec(compile(ast.Module(selected, []), str(SOURCE), "exec"), ns)
    runtime.configs = ns['get_autotune_config']() if INT8 else []
    return ns['matmul'], runtime


def pair(m=3, k=None, n=5, **kwargs):
    k = (512 if INT8 else 35) if k is None else k
    return Tensor((m,k), **kwargs), Tensor((k//4 if INT8 else k,n), dtype=torch.uint8 if INT8 else torch.float32)


def main():
    original = ast.parse(ORIGINAL.read_text().split("################################################################")[0]).body
    candidate = ast.parse(SOURCE.read_text()).body
    production=copy.deepcopy(candidate[:len(original)])
    host=next(n for n in production if isinstance(n,ast.FunctionDef) and n.name=='matmul')
    address_guards=[n for n in host.body if isinstance(n,ast.If) and 'formed output byte pointers' in ast.unparse(n)]
    assert len(address_guards)==1
    host.body.remove(address_guards[0])
    contexts=[n for n in host.body if isinstance(n,ast.With)]
    assert len(contexts)==1 and len(contexts[0].items)==1 and len(contexts[0].body)==1
    assert ast.dump(contexts[0].items[0].context_expr)==ast.dump(ast.parse('torch.cuda.device(a.device)',mode='eval').body)
    assert contexts[0].items[0].optional_vars is None
    host.body[host.body.index(contexts[0]):host.body.index(contexts[0])+1]=contexts[0].body
    assert [ast.dump(n) for n in original] == [ast.dump(n) for n in production]
    assert isinstance(candidate[-1], ast.FunctionDef) and candidate[-1].name == 'matmul'
    assert ast.dump(candidate[-1].body[-1].value) == ast.dump(ast.parse('_unchecked_matmul(a, b)' if INT8 else '_unchecked_matmul(a, b, activation)',mode='eval').body)
    admissions = [pair(), pair(m=0), pair(n=0), pair(k=0), pair(m=1,n=1), pair(m=33,n=65), pair(k=128 if INT8 else 1)]
    a,b = pair(m=1)
    a._strides = (12345,1)
    admissions.append((a,b))
    a,b = pair(); b._strides=(1, b.shape[0]+5); admissions.append((a,b))
    a,b = pair(); b._strides=(0,0); admissions.append((a,b))
    a,b = pair(); a.data=torch.tensor([-128.999,127.999,-0.0,0.999]); admissions.append((a,b))
    if not INT8:
        a,b = pair(); a.data=torch.tensor([float('nan'),float('inf')]); admissions.append((a,b))
        a,b = pair(device=1); b.device=SimpleNamespace(index=1); admissions.append((a,b))
    for a,b in admissions:
        for activation in ([''] if INT8 else ['', 'leaky_relu', 'identity', 'LEAKY_RELU']):
            call,rt = instantiate(); rt.activation=activation
            result=call(a,b) if INT8 else call(a,b,activation)
            assert result is rt.output and rt.events == ['allocation','launch']
            assert rt.current_device == 0 and rt.launch_devices == [a.device.index]
    # Foreign unaligned FP32 storage is rejected; empty tensors and byte B need no alignment.
    a,b=pair(k=0); a.pointer=1; b.pointer=1
    call,rt=instantiate(); rt.activation=""
    call(a,b)
    if INT8:
        a,b=pair(); b.pointer=1
        call,rt=instantiate(); call(a,b)
    rejected=[]
    rejected.append((object(),pair()[1],TypeError))
    for which in ([0] if INT8 else [0,1]):
        a,b=pair(); (a,b)[which].pointer=4097; rejected.append((a,b,ValueError))
    for attr,val,err in [('layout',torch.sparse_coo,ValueError),('ndim',1,ValueError),('dtype',torch.float16,TypeError),('is_cuda',False,ValueError),('_contiguous',False,ValueError),('_negative',True,ValueError),('_strides',(-1,1),ValueError),('_strides',(2**31,1),ValueError)]:
        a,b=pair(); setattr(a,attr,val); rejected.append((a,b,err))
    a,b=pair(); b.device=SimpleNamespace(index=1); rejected.append((a,b,ValueError))
    a,b=pair(); b._negative=True; rejected.append((a,b,ValueError))
    a,b=pair(); b.shape=(b.shape[0]+1,b.shape[1]); rejected.append((a,b,ValueError))
    a,b=pair(); a._strides=(2**30,1); rejected.append((a,b,ValueError))
    a,b=pair(); b._strides=(2**30,1); rejected.append((a,b,ValueError))
    a,b=pair(m=2**31); rejected.append((a,b,ValueError))
    if INT8:
        a,b=pair(device=1); b.device=SimpleNamespace(index=1); rejected.append((a,b,ValueError))
        for values in [[float('nan')],[float('inf')],[-float('inf')],[-129.0],[128.0]]:
            a,b=pair(); a.data=torch.tensor(values); rejected.append((a,b,ValueError))
    for which in [0,1]:
        a,b=pair(); (a,b)[which].pointer=2**64-4; rejected.append((a,b,ValueError))
    for a,b,error in rejected:
        call,rt=instantiate()
        try: call(a,b)
        except error: pass
        else: raise AssertionError('unsupported input was accepted')
        assert rt.events == []
    if not INT8:
        call,rt=instantiate()
        for activation in [None,0,False]:
            try: call(*pair(),activation)
            except TypeError: pass
            else: raise AssertionError('non-string activation accepted')
        assert rt.events == []
    for failure in ['allocation','launch']:
        call,rt=instantiate(failure); rt.activation=''
        a,b=pair(device=1); b.device=SimpleNamespace(index=1)
        try: call(a,b)
        except RuntimeError as error: assert str(error)==failure
        else: raise AssertionError('runtime failure swallowed')
        assert rt.current_device==0 and rt.launch_devices==([] if failure=='allocation' else [1])
    call,rt=instantiate('outputAddress');rt.activation=''
    try: call(*pair())
    except ValueError: pass
    else: raise AssertionError('output byte envelope was unchecked')
    assert rt.events==['allocation']
    # Actual CPU logical-negative views retain the raw pointer; rejection is needed.
    real=torch.ones((1,512),dtype=torch.float32)
    negative=torch._neg_view(real)
    assert negative.is_contiguous() and negative.is_neg() and negative.data_ptr()==real.data_ptr()
    assert torch.equal(negative,-real)
    print(f'PASS {TASK}: unchanged kernel/config AST and explicit post-allocation byte guard; {len(admissions)} admission families; {len(rejected)} rejections; exact allocation/launch/return and propagated failures. No GPU execution.')
    print('PASS: owning-device context preserves production AST after exact context unwrapping; cuda:1 success and submission exception restore ambient cuda:0.')

if __name__ == '__main__': main()
