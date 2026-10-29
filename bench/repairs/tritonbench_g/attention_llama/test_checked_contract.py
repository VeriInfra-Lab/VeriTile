"""Executed host guards against instrumented tensor observations; no GPU claim."""
import ast
import importlib.util
import json
import math
import sys
import struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'attention_llama_repaired.py'
spec=importlib.util.spec_from_file_location('llama_checked_host',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
records=[];allocations=[];launches=[]
current_device,launch_devices,fail_launch=['cuda:0'],[],[False]
class DeviceContext:
    def __init__(self,device):self.device=device
    def __enter__(self):self.previous=current_device[0];current_device[0]=self.device
    def __exit__(self,*exc):current_device[0]=self.previous
class Tensor:
    next_base=2**32
    def __init__(self,shape=(1,2,65,16),dtype='fp16',strides=None,base=None):
        self.shape=tuple(shape);self.dtype=dtype;self.ndim=len(shape)
        self.strides=tuple(strides) if strides is not None else tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
        self.layout='strided';self.device='cuda:0';self.is_cuda=True;self.neg=False;self.conj=False
        self.base=Tensor.next_base if base is None else base;Tensor.next_base+=2**24
        self.bytes=np.zeros(min(math.prod(shape),1<<20),dtype=np.int8)
        self.observations=0
    def is_neg(self):return self.neg
    def is_conj(self):return self.conj
    def stride(self,i=None):return self.strides if i is None else self.strides[i]
    def numel(self):return math.prod(self.shape)
    def size(self,i=None):return self.shape if i is None else self.shape[i]
    def data_ptr(self):return self.base
    def element_size(self):return 1 if self.dtype=='int8' else 2
    def __and__(self,mask):
        self.observations+=1
        return self.bytes&mask

def empty_like(q):
    o=Tensor(q.shape,q.dtype);o.device=q.device;allocations.append(o);return o
class Spy:
    def __getitem__(self,grid):
        def launch(*args,**kw):
            assert current_device[0]==args[0].device
            launch_devices.append(current_device[0])
            if fail_launch[0]:raise RuntimeError('mock submission failure')
            launches.append((grid,args,kw))
        return launch
m.torch=SimpleNamespace(Tensor=Tensor,strided='strided',float16='fp16',int8='int8',empty_like=empty_like,cuda=SimpleNamespace(device=DeviceContext))
m._fwd_kernel=Spy()
def inputs(H=3,N=65,D=16,fp8=False,B=1,NH=2):
    return [Tensor((B,NH,H,D)),Tensor((B,NH,N,D),'int8' if fp8 else 'fp16'),Tensor((B,NH,N,D),'int8' if fp8 else 'fp16'),.25,False,0]
def check(name,args):
    allocations.clear();launches.clear()
    before=[t.bytes.copy() for t in args[:3]]
    out=m.triton_fa(*args)
    assert out is allocations[0] and len(allocations)==1
    assert len(launches)==bool(args[0].numel())
    if launches:
        grid,actual,kw=launches[0]
        assert actual[:4]==tuple(args[:4]) and actual[4] is out
        assert grid==((args[0].shape[2]+63)//64,args[0].shape[0]*args[0].shape[1])
        assert actual[5:21]==tuple(s for t in (*args[:3],out) for s in t.stride())
        assert kw['num_stages']==2 and kw['num_warps']==(4 if args[0].shape[-1]<=64 else 8)
        assert kw['USE_FP8']==(args[1].dtype=='int8')
        assert kw['IS_CAUSAL'] is args[4] and kw['start_position']==args[5]
    for t,old in zip(args[:3],before):assert np.array_equal(t.bytes,old)
    records.append(dict(case=name,status='PASS'))
def reject(name,args):
    allocations.clear();launches.clear()
    try:m.triton_fa(*args)
    except (ValueError,TypeError):pass
    else:raise AssertionError(name)
    assert not launches
    records.append(dict(case=name,status='PASS'))
for D in (16,32,64,128):
    for fp8 in (False,True):check('dtype-head-profile',inputs(D=D,fp8=fp8))
for H in (0,1,63,64,65):
    a=inputs(H=H);a[4]=True;check('causal-tail',a)
for dim in ('batch','head'):
    check('empty-'+dim,inputs(B=0 if dim=='batch' else 1,NH=0 if dim=='head' else 2))
for scale in (0,-.5,2,1e-40):
    a=inputs();a[3]=scale;check('preserved-scale',a)
# Preserve the actual Python binary64 scalar bits through the executed caller.
# The recorded fp32 ABI conversion is checked independently of its Real projection.
for scale, abi_bits in ((0.0,0x00000000),(-0.0,0x80000000),(1e-50,0x00000000),(-1e-50,0x80000000)):
    a=inputs();a[3]=scale;check('scalar-signed-zero-or-underflow',a)
    passed=launches[0][1][3]
    assert struct.pack('>d',passed)==struct.pack('>d',scale)
    assert struct.unpack('>I',struct.pack('>f',passed))[0]==abi_bits
for scale in (1e-40,-1e-40):
    a=inputs();a[3]=scale;check('scalar-signed-subnormal',a)
    passed=launches[0][1][3]
    assert struct.pack('>d',passed)==struct.pack('>d',scale)
    bits=struct.unpack('>I',struct.pack('>f',passed))[0]
    assert bits&0x7fffffff and bool(bits>>31)==(scale<0)
a=inputs();a[2]=a[1];check('input-alias',a)
a=inputs();a[0].strides=(0,0,0,0);check('broadcast-input',a)
a=inputs(fp8=True);a[1].bytes[0]=-128;check('finite-negative-zero',a)
assert a[1].observations==1 and a[2].observations==1
# All 256 encodings exercise the actual bitwise-any-item branch.
for value in range(-128,128):
    a=inputs(H=1,N=1,fp8=True,NH=1);a[1].bytes[0]=value
    if value&124==124:reject('reject-e5m2-'+str(value),a)
    else:check('admit-e5m2-'+str(value),a)
a=inputs(fp8=True);a[2].bytes[-1]=127;reject('value-content-check',a)
for which,field,value in ((0,'dtype','fp32'),(1,'dtype','fp32'),(0,'is_cuda',False),(1,'device','cuda:1'),(0,'neg',True),(2,'conj',True),(1,'layout','sparse')):
    a=inputs();setattr(a[which],field,value);reject(field,a)
for idx,value in ((3,float('nan')),(3,float('inf')),(3,True),(3,10**500),(4,1),(5,-1),(5,2**31),(5,True)):
    a=inputs();a[idx]=value;reject('scalar-'+str(idx),a)
a=inputs();a[4]=True;a[5]=1;reject('causal-position',a)
a=inputs(N=0);reject('empty-key',a)
a=inputs(D=24);reject('unsupported-head',a)
a=inputs();a[0].strides=(-1,1,1,1);reject('negative-stride',a)
a=inputs();a[1].shape=(1,2,65,32);reject('head-mismatch',a)
a=inputs();a[2].shape=(1,2,66,16);reject('kv-shape',a)
a=inputs();a[0].ndim=3;reject('rank',a)
a=inputs();a[0].base+=1;reject('natural-alignment',a)
a=inputs();a[1].base=2**64-2;reject('absolute-address-overflow',a)
a=inputs();a[0].strides=(2**31,1,1,1);reject('formed-stride-overflow',a)
a=inputs(B=65536,NH=1);reject('launch-axis',a)
# Output addresses are checked after allocation too.
ordinary_empty=m.torch.empty_like
def bad_output(q):
    o=ordinary_empty(q);o.base=2**64-2;return o
m.torch.empty_like=bad_output
reject('allocated-output-address',inputs())
m.torch.empty_like=ordinary_empty
# Exact kernel AST delta preserves all precision statements and original branches.
original=HERE.parents[3]/'bench/tritonbench_g/attention_llama/attention_llama.py'
def kernel(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_fwd_kernel')
old,new=kernel(original),kernel(SOURCE)
loop=next(n for n in old.body if isinstance(n,ast.For))
mask=next(n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='where')
loop.body.remove(mask)
mask.value.args[0].left.value.id='block_n_offs'
scale=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.AugAssign) and isinstance(n.op,ast.Mult) and isinstance(n.target,ast.Name) and n.target.id=='qk')
loop.body.insert(scale+1,mask)
assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
records.append(dict(case='exact-kernel-repair-delta',status='PASS'))
a=inputs()
for t in a[:3]:t.device='cuda:1'
check('owning_device_mismatch',a)
assert current_device[0]=='cuda:0' and launch_devices[-1]=='cuda:1'
fail_launch[0]=True
try:m.triton_fa(*a)
except RuntimeError:pass
else:raise AssertionError('expected mock submission failure')
finally:fail_launch[0]=False
assert current_device[0]=='cuda:0' and launch_devices[-1]=='cuda:1'
records.append(dict(case='owning_device_exception_restoration',status='PASS'))
print(json.dumps(dict(cases=len(records),all_byte_encodings=256,actual_host_helpers=True,gpu_execution=False)))
