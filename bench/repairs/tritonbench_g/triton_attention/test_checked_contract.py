"""Actual forward/backward host code with allocation, copy and kernel spies."""
import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'Whole001/triton_attention_repaired.py'
spec=importlib.util.spec_from_file_location('triton_attention_checked',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device=SimpleNamespace(type='cuda',index=0)
events=[];launches=[];allocations=[];records=[]
current_device=99
class DeviceContext:
    def __init__(self,selected):self.selected=selected.index
    def __enter__(self):
        global current_device
        self.before=current_device;current_device=self.selected;events.append(("device-enter",self.selected))
    def __exit__(self,*args):
        global current_device
        current_device=self.before;events.append(("device-exit",self.before))
class Tensor:
    nextbase=2**32
    def __init__(self,shape,dtype='fp32',strides=None):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device
        self.layout='strided';self.neg=False;self.conj=False;self.contig=True
        self.strides=tuple(strides) if strides is not None else tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
        self.storagebase=Tensor.nextbase;Tensor.nextbase+=2**24
        self.offset=17;self.pointer=self.storagebase+self.element_size()*self.offset
        self.nbytes=self.element_size()*(self.offset+self.numel());self.valueToken=object()
    def numel(self):return math.prod(self.shape)
    def stride(self,i=None):return self.strides if i is None else self.strides[i]
    def is_contiguous(self):return self.contig
    def is_conj(self):return self.conj
    def is_neg(self):return self.neg
    def data_ptr(self):return self.pointer
    def storage_offset(self):return self.offset
    def element_size(self):return 2 if self.dtype=='fp16' else 4
    def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.storagebase,nbytes=lambda:self.nbytes)
    def contiguous(self):
        events.append(('contiguous',self))
        if self.contig:return self
        out=Tensor(self.shape,self.dtype);out.valueToken=self.valueToken;events.append(('copied',out));return out
    def resolve_neg(self):raise AssertionError('unexpected normalization')
def empty(shape,device,dtype):
    t=Tensor(shape,dtype);t.device=device;allocations.append(t);events.append(('empty',t));return t
def empty_like(t,dtype=None):return empty(t.shape,t.device,t.dtype if dtype is None else dtype)
def zeros_like(t,dtype=None):
    out=empty_like(t,dtype);events[-1]=('zeros',out);return out
class Spy:
    def __init__(self,name):self.name=name
    def __getitem__(self,grid):
        def launch(*args,**kw):
            assert current_device==args[0].device.index
            launches.append((self.name,grid,args,kw));events.append((self.name,args))
        return launch
m.torch=SimpleNamespace(Tensor=Tensor,strided='strided',float32='fp32',float16='fp16',
                        empty=empty,empty_like=empty_like,zeros_like=zeros_like,cuda=SimpleNamespace(device=DeviceContext))
m._fwd_kernel=Spy('forward');m._bwd_preprocess=Spy('preprocess');m._bwd_kernel=Spy('backward')
def context():
    ctx=SimpleNamespace()
    def save(*tensors):ctx.saved_tensors=tensors
    ctx.save_for_backward=save;return ctx
def inputs(B=1,H=2,N=128,D=16,dtype='fp32'):return [Tensor((B,H,N,D),dtype) for _ in range(3)]
def success(name,a,scale=.125,noncontig=False,compat=False):
    events.clear();launches.clear();allocations.clear();ctx=context()
    entry=m._attention_compatibility if compat else m._attention
    out=entry.forward(ctx,*a,scale)
    assert out is allocations[0] and len(allocations)==3 and current_device==99
    assert ctx.saved_tensors==(*a,out,allocations[1],allocations[2]) and ctx.sm_scale is scale
    assert launches[0][2][:4]==(*a,scale) and launches[0][2][6] is out
    gradient=Tensor(out.shape,out.dtype)
    if noncontig:gradient.contig=False;gradient.strides=tuple(2*s for s in gradient.strides)
    result=entry.backward(ctx,gradient)
    assert current_device==99
    assert len(result)==4 and result[3] is None and result[:3]==tuple(allocations[3:6])
    assert result[0].dtype=='fp32' and result[1].dtype==a[1].dtype and result[2].dtype==a[2].dtype
    assert [x[0] for x in launches]==['forward','preprocess','backward']
    stages=[x[0] for x in events]
    assert stages.index('contiguous')<stages.index('zeros')<stages.index('preprocess')<stages.index('backward')
    copied=next((x[1] for x in events if x[0]=='copied'),gradient)
    assert launches[1][2][1] is copied and copied.valueToken is gradient.valueToken
    assert launches[2][2][3] is scale and launches[2][2][6:9]==result[:3]
    assert launches[0][3]['num_stages']==2 and launches[2][3]['num_stages']==1
    assert launches[0][1]==(a[0].shape[2]//128,a[0].shape[0]*a[0].shape[1],1)
    if type(scale) is float:assert struct.pack('>d',launches[0][2][3])==struct.pack('>d',scale)
    records.append(name)
def reject_forward(name,a,scale=.125):
    launches.clear()
    try:m._attention.forward(context(),*a,scale)
    except (TypeError,ValueError,OverflowError,AssertionError):pass
    else:raise AssertionError(name)
    assert not launches;records.append(name)
for d in (16,32,64,128):
    success('fp32-domain',inputs(D=d))
    success('gradient-original-copy',inputs(D=d),noncontig=True)
    success('explicit-fp16-compatibility',inputs(D=d,dtype='fp16'),compat=True)
success('multi-batch-head',inputs(B=2,H=3,N=384))
for scale in (1,2,1.,1e-40,3e38):success('raw-scale',inputs(),scale=scale)
a=inputs();a[2]=a[0];success('read-alias',a)
a=inputs(B=1,H=1);ctx=context();out=m._attention.forward(ctx,*a,.125)
gradient=Tensor(out.shape,strides=(7,13,16,1));m._attention.backward(ctx,gradient)
assert launches[-2][2][1] is gradient and current_device==99
records.append('gradient-singleton-stride-identity')
for dtype in ('fp16','bf16'):reject_forward('checked-dtype-domain',inputs(dtype=dtype))
for idx,field,value in [(0,'layout','sparse'),(1,'neg',True),(2,'conj',True),(0,'contig',False),
                       (0,'ndim',3),(0,'pointer',1),(1,'pointer',2**64-4),(2,'nbytes',1)]:
    a=inputs();setattr(a[idx],field,value);reject_forward(field,a)
a=inputs();a[1].shape=(1,2,256,16);reject_forward('shape-mismatch',a)
a=inputs();a[0].strides=(1,2048,16,1);reject_forward('singleton-stride',a)
for b,h,n,d in [(0,2,128,16),(1,0,128,16),(1,2,0,16),(1,2,129,16),(1,2,128,15),(1,65536,128,16),(2**20,1,128,128)]:
    reject_forward('shape-bound',inputs(b,h,n,d))
for scale in (0,-0.,-1,1e-50,math.inf,math.nan,1e39,True,'1'):
    reject_forward('scale-bound',inputs(),scale)
for field,value in [('dtype','fp16'),('shape',(1,2,256,16)),('conj',True),('neg',True),('pointer',1),('nbytes',1)]:
    a=inputs();ctx=context();out=m._attention.forward(ctx,*a,.125);gradient=Tensor(out.shape)
    setattr(gradient,field,value);launches.clear()
    try:m._attention.backward(ctx,gradient)
    except (TypeError,ValueError,AssertionError):pass
    else:raise AssertionError(field)
    assert not launches;records.append('backward-'+field)
nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_fwd_kernel','_bwd_preprocess','_bwd_kernel')]
digest=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
assert digest=='7919e8651fef50ea9a4827d57a87e768ae4d2cdb0bd5f12262841754c195e93d'
original=HERE.parents[3]/'bench/tritonbench_g/triton_attention/triton_attention.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='3eb3038c22a9816c1cef15262c8f0b1adcf4f2f960b1235859211f8ed81a1a38'
records.append('original-and-kernel-AST-preserved')
print(json.dumps(dict(cases=len(records),actual_helpers=True,owning_device_restored=True,fp32_checked_fp16_separate=True,original_gradient_copy=True,gpu_execution=False)))
