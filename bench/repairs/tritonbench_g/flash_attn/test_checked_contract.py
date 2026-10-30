"""Actual host helpers and allocation/launch branches with ordinary observations."""
import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'flash_attn_repaired.py'
spec = importlib.util.spec_from_file_location('flash_checked_host', SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
allocations = []; launches = []; contexts = []; records = []
current_device = 'cuda:5'
class Tensor:
    nextbase = 2**32
    def __init__(self, shape, strides=None, base=None):
        self.shape = tuple(shape); self.ndim = len(shape)
        self.strides = tuple(strides) if strides is not None else tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
        self.storagebase = Tensor.nextbase if base is None else base; Tensor.nextbase += 2**24
        self.offset = 17; self.pointer = self.storagebase + 4*self.offset
        self.nbytes = 4*(self.offset + 1 + sum((n-1)*s for n,s in zip(self.shape,self.strides)))
        self.dtype = 'fp32'; self.layout = 'strided'; self.device = 'cuda:0'; self.is_cuda = True
        self.requires_grad = False; self.neg = False; self.conj = False
    def is_neg(self): return self.neg
    def is_conj(self): return self.conj
    def stride(self, i=None): return self.strides if i is None else self.strides[i]
    def data_ptr(self): return self.pointer
    def storage_offset(self): return self.offset
    def untyped_storage(self): return SimpleNamespace(data_ptr=lambda:self.storagebase, nbytes=lambda:self.nbytes)
    def contiguous(self): raise AssertionError('unexpected copy')
    def resolve_neg(self): raise AssertionError('unexpected normalization')
def empty(shape, device, dtype):
    t = Tensor(shape); t.device = device; t.dtype = dtype; allocations.append(t); return t
def empty_like(t): return empty(t.shape, t.device, t.dtype)
class Device:
    def __init__(self, target): self.target = target
    def __enter__(self):
        global current_device
        self.previous = current_device; current_device = self.target; contexts.append(('enter', self.target))
    def __exit__(self, *args):
        global current_device
        current_device = self.previous; contexts.append(('exit', self.previous))
class Spy:
    def __getitem__(self, grid):
        def launch(*args, **kw): launches.append((grid, args, kw, current_device))
        return launch
m.torch = SimpleNamespace(Tensor=Tensor, strided='strided', float32='fp32', empty=empty,
                         empty_like=empty_like, cuda=SimpleNamespace(device=Device))
m._fwd_kernel = Spy()
def inputs(B=1,H=2,N=65,D=32): return [Tensor((B,H,N,D)) for _ in range(3)]
def success(name, a, causal=True, scale=1):
    allocations.clear(); launches.clear(); contexts.clear()
    out = m.flash_attn_triton(*a, causal=causal, sm_scale=scale)
    assert out is allocations[0] and len(allocations)==2
    assert out.shape==a[0].shape and allocations[1].shape==(a[0].shape[0]*a[0].shape[1],a[0].shape[2])
    grid, actual, kw, device = launches[0]
    assert actual[:3]==tuple(a) and actual[3] is scale and actual[4] is allocations[1] and actual[5] is out
    assert actual[6:22]==tuple(s for t in (*a,out) for s in t.stride())
    assert grid==((a[0].shape[2]+127)//128,a[0].shape[0]*a[0].shape[1],1)
    assert kw['num_warps']==(4 if a[0].shape[3]<=64 else 8) and kw['num_stages']==4
    assert kw['IS_CAUSAL'] is causal and device==a[0].device and current_device=='cuda:5'
    if type(scale) is float: assert struct.pack('>d',actual[3])==struct.pack('>d',scale)
    records.append(name)
def reject(name,a,causal=True,scale=1):
    launches.clear()
    try: m.flash_attn_triton(*a,causal=causal,sm_scale=scale)
    except (TypeError,ValueError,AssertionError,OverflowError): pass
    else: raise AssertionError(name)
    assert not launches and current_device=='cuda:5'; records.append(name)
for causal in (False,True):
    for d in (16,32,64,128): success('shape-profile',inputs(D=d),causal)
for n in (1,63,64,65,127,128,129): success('row-tail',inputs(N=n))
for scale in (0,1,-1,0.0,-0.0,1e-50,-1e-50,1e-40,-1e-40,3.4028234663852886e38):
    success('raw-scalar',inputs(),scale=scale)
a=inputs(B=2,H=3)
a[0]=Tensor(a[0].shape,(200000,50000,511,2)); a[1]=Tensor(a[1].shape,(300000,70000,301,3))
a[2]=Tensor(a[2].shape,(400000,90000,401,1)); success('independent-strides',a)
for i,field,value in [(0,'dtype','fp16'),(1,'layout','sparse'),(2,'is_cuda',False),(1,'device','cuda:1'),
                       (0,'requires_grad',True),(1,'neg',True),(2,'conj',True),(0,'ndim',3),
                       (0,'offset',-1),(1,'nbytes',1),(0,'pointer',1),(2,'pointer',2**64-4)]:
    a=inputs(); setattr(a[i],field,value); reject(field,a)
a=inputs(); a[1].shape=(1,2,66,32);reject('shape-mismatch',a)
a=inputs(); a[1].storagebase=a[0].storagebase;reject('shared-storage',a)
for strides in [(1,1,1,1),(100000,50000,32,0),(100000,50000,-32,1),(2**31,50000,32,1)]:
    a=inputs();a[0].strides=strides;reject('stride',a)
for shape in [(0,2,65,32),(1,2,0,32),(1,2,65,15),(1,65536,1,16),(1,1,2**31,16)]:
    reject('shape-bound',[Tensor(shape) for _ in range(3)])
for scale in (math.inf,-math.inf,math.nan,3.5e38,True,'1'):
    reject('bad-scale',inputs(),scale=scale)
reject('bad-causal',inputs(),causal=1)
normal_empty=m.torch.empty
def bad_scratch(shape,device,dtype):
    t=normal_empty(shape,device,dtype)
    if len(shape)==2:t.pointer=2**64-4
    return t
m.torch.empty=bad_scratch
reject('scratch-pointer',inputs());m.torch.empty=normal_empty
normal_output=m.torch.empty_like
def bad_output(t):
    o=normal_output(t);o.pointer+=1;return o
m.torch.empty_like=bad_output
reject('output-pointer',inputs());m.torch.empty_like=normal_output
def kernel_digest(path):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_fwd_kernel')
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
assert kernel_digest(SOURCE)=='15ecd16b6ea2d715e06e9037c39345fe35977153b34d23e29057bd8f31fdad8d'
original=HERE.parents[3]/'bench/tritonbench_g/flash_attn/flash_attn.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='fedc0f296ff2f6c4176ee990c1ea4a29129466998c8ac2fc3f6b1ccc5dfc84a1'
records.append('kernel-AST-and-original-preserved')
print(json.dumps(dict(cases=len(records),actual_helpers=True,raw_scalar_preserved=True,device_restored=True,gpu_execution=False)))
