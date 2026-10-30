"""Executed source host checks with instrumented tensor/runtime observations."""
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'attn_fwd_triton_repaired.py'
spec=importlib.util.spec_from_file_location('attn_checked_host',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
allocations=[];launches=[];records=[]
current_device,launch_devices,fail_launch=['cuda:0'],[],[False]
class DeviceContext:
    def __init__(self,device):self.device=device
    def __enter__(self):self.previous=current_device[0];current_device[0]=self.device
    def __exit__(self,*exc):current_device[0]=self.previous
class Tensor:
    nextbase=2**32
    def __init__(self,shape,dtype='fp32',strides=None,base=None):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype
        self.strides=tuple(strides) if strides is not None else tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
        self.base=Tensor.nextbase if base is None else base;Tensor.nextbase+=2**24
        self.layout='strided';self.device='cuda:0';self.is_cuda=True;self.neg=False;self.conj=False;self.contig=True
    def is_neg(self):return self.neg
    def is_conj(self):return self.conj
    def is_contiguous(self):return self.contig
    def numel(self):return math.prod(self.shape)
    def stride(self,i=None):return self.strides if i is None else self.strides[i]
    def data_ptr(self):return self.base
    def element_size(self):return 2 if self.dtype=='bf16' else 4
    def resolve_neg(self):raise AssertionError('unexpected normalization')
    def contiguous(self):raise AssertionError('unexpected packing')
    def view(self,*args):raise AssertionError('unexpected flatten copy/view')
def empty(shape,device,dtype):
    t=Tensor(shape,dtype);t.device=device;allocations.append(t);return t
def empty_like(t,dtype):return empty(t.shape,t.device,dtype)
class Spy:
    def __getitem__(self,grid):
        def launch(*args,**kw):
            assert current_device[0]==args[0].device
            launch_devices.append(current_device[0])
            if fail_launch[0]:raise RuntimeError('mock submission failure')
            launches.append((grid,args,kw))
        return launch
m.torch=SimpleNamespace(Tensor=Tensor,strided='strided',float32='fp32',bfloat16='bf16',empty=empty,empty_like=empty_like,cuda=SimpleNamespace(device=DeviceContext))
m._attn_fwd_repaired=Spy()
def inputs(B=1,H=2,N=65,D=96):
    return [Tensor((B,H,N,D)) for _ in range(3)]+[Tensor((B,H,(N+127)//128)),Tensor((B,H,(N+63)//64))]
def success(name,a):
    allocations.clear();launches.clear()
    out=m.forward_repaired(*a)
    assert out is allocations[0] and out.dtype=='bf16' and out.shape==a[0].shape
    assert len(allocations)==2 and allocations[1].shape==a[0].shape[:3] and allocations[1].dtype=='fp32'
    assert len(launches)==bool(a[0].numel())
    if launches:
        grid,actual,kw=launches[0]
        assert actual[:5]==tuple(a) and actual[5] is out
        assert actual[6:22]==tuple(s for t in (*a[:3],out) for s in t.stride())
        assert kw['STAGE']==3 and kw['num_warps']==8 and kw['num_stages']==3
        assert kw['BLOCK_M']==128 and kw['BLOCK_N']==64
        assert kw['BLOCK_DMODEL']==max(16,1<<(a[0].shape[3]-1).bit_length())
        assert grid==((a[0].shape[2]+127)//128,a[0].shape[0]*a[0].shape[1],1)
    records.append(dict(case=name,status='PASS'))
def reject(name,a):
    allocations.clear();launches.clear()
    try:m.forward_repaired(*a)
    except (ValueError,TypeError,AssertionError):pass
    else:raise AssertionError(name)
    assert not launches;records.append(dict(case=name,status='PASS'))
for n,d in [(1,1),(1,16),(3,31),(63,96),(64,128),(65,129),(127,16),(128,31),(129,96)]:success('tail-head',inputs(N=n,D=d))
for B,H,N,D in ((0,2,4,16),(1,0,4,16),(1,2,0,16),(1,2,4,0)):
    a=inputs(B,H,N,D);a[3].shape=(0,);a[4].shape=(0,);a[3].contig=False;success('empty-keeps-M',a)
a=inputs();a[2]=a[1];success('read-alias',a)
a=inputs();a[0].strides=(0,0,0,0);success('zero-strides',a)
a=inputs();a[0].strides=(39999,7777,251,2);a[1].strides=(29999,5111,301,3);success('independent-strides',a)
for idx,field,val in ((0,'dtype','bf16'),(3,'dtype','fp16'),(0,'device','cpu'),(0,'is_cuda',False),(1,'device','cuda:1'),(0,'neg',True),(4,'conj',True),(2,'layout','sparse'),(0,'ndim',3)):
    a=inputs();setattr(a[idx],field,val);reject(field,a)
for idx in (3,4):
    a=inputs();a[idx].contig=False;reject('unpacked-scales',a)
    a=inputs();a[idx].shape=(0,);reject('short-scales',a)
a=inputs();a[0].strides=(-1,1,1,1);reject('negative-stride',a)
a=inputs();a[1].shape=(1,2,66,96);reject('shape-mismatch',a)
a=inputs();a[2].base+=1;reject('natural-alignment',a)
a=inputs();a[4].base=2**64-4;reject('absolute-pointer',a)
a=inputs();a[0].strides=(2**63,1,1,1);reject('signed64-formed',a)
reject('grid-y',inputs(B=65536,H=1));reject('grid-x',inputs(N=128*(2**31-1)+1));reject('tile-bound',inputs(D=1048577))
original_empty=m.torch.empty_like
def bad_output(t,dtype):
    out=original_empty(t,dtype);out.base=2**64-2;return out
m.torch.empty_like=bad_output;reject('output-pointer',inputs());m.torch.empty_like=original_empty
# The two production kernels remain exactly the accepted repair graph.
tree=ast.parse(SOURCE.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_attn')]
digest=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
EXPECTED='ec0641d6097d712953161a9fbdd972dcfd12334bd324361e26dbc06d256ab1c4'
assert digest==EXPECTED
original=HERE.parents[3]/'bench/tritonbench_g/attn_fwd_triton/attn_fwd_triton.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='de0fb466adb7844b34f53e44225c25411faad342e4192e8f9d5bbe71f154df12'
records.append(dict(case='preserved-kernel-AST-and-original',status='PASS'))
a=inputs()
for t in a:t.device='cuda:1'
success('owning-device-mismatch',a)
assert current_device[0]=='cuda:0' and launch_devices[-1]=='cuda:1'
fail_launch[0]=True
try:m.forward_repaired(*a)
except RuntimeError:pass
else:raise AssertionError('expected mock submission failure')
finally:fail_launch[0]=False
assert current_device[0]=='cuda:0' and launch_devices[-1]=='cuda:1'
records.append(dict(case='owning-device-exception-restoration',status='PASS'))
print(json.dumps(dict(cases=len(records),actual_helpers=True,all_original_allocations=True,gpu_execution=False)))
