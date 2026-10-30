"""Actual checked host branches, retained copies, allocation and launch controls."""
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
SOURCE=HERE/'Whole001/chunk_retention_repaired.py'
spec=importlib.util.spec_from_file_location('retention_checked',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device=SimpleNamespace(type='cuda',index=2)
events=[];allocations=[];launches=[];records=[];current=17
class Tensor:
    base=2**32
    def __init__(self,shape,dtype='fp32',strides=None):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device;self.layout='strided';self.is_cuda=True
        self.neg=False;self.conj=False;self.contig=True;self.strides=strides or tuple(math.prod(max(x,1) for x in shape[i+1:]) for i in range(len(shape)))
        self.token=object();self.base=Tensor.base;Tensor.base+=2**26;self.offset=13;self.pointer=self.base+4*self.offset
        self.nbytes=4*(self.offset+self.numel()+7)
    def numel(self):return math.prod(self.shape)
    def stride(self,i=None):return self.strides if i is None else self.strides[i]
    def element_size(self):return 4 if self.dtype=='fp32' else 2
    def storage_offset(self):return self.offset
    def data_ptr(self):return self.pointer
    def untyped_storage(self):return SimpleNamespace(nbytes=lambda:self.nbytes,data_ptr=lambda:self.base)
    def is_contiguous(self):return self.contig
    def is_conj(self):return self.conj
    def is_neg(self):return self.neg
    def contiguous(self):
        events.append(('contiguous',self))
        if self.contig:return self
        out=Tensor(self.shape,self.dtype);out.token=self.token;events.append(('copy',self,out));return out
    def detach(self):events.append(('detach',self));return self
    def new_empty(self,*shape,**kw):return empty(shape,self.device,kw.get('dtype',self.dtype))
    def to(self,dtype):assert dtype==self.dtype;events.append(('to',self));return self
    def sum(self,axis):
        assert axis==0;out=empty(self.shape[1:],self.device,self.dtype);events.append(('sum',self,out));return out
class Device:
    def __init__(self,dev):self.dev=dev.index
    def __enter__(self):
        global current
        self.old=current;current=self.dev
    def __exit__(self,*args):
        global current
        current=self.old
class Spy:
    def __init__(self,name):self.name=name
    def __getitem__(self,grid):
        def call(*args,**kw):
            assert current==args[0].device.index
            launches.append((self.name,grid,args,kw));events.append((self.name,args))
        return call
def empty(shape,dev,dtype):
    out=Tensor(shape,dtype);out.device=dev;allocations.append(out);events.append(('empty',out));return out
def empty_like(t):return empty(t.shape,t.device,t.dtype)
def zeros_like(t):
    out=empty_like(t);events.append(('zeros',out));return out
m.torch=SimpleNamespace(Tensor=Tensor,strided='strided',float32='fp32',empty_like=empty_like,zeros_like=zeros_like,cuda=SimpleNamespace(device=Device))
for suffix in ('fwd_kernel_h','fwd_kernel_o','bwd_kernel_dh','bwd_kernel_dqkv'):
    setattr(m,'chunk_retention_'+suffix,Spy(suffix))
def ctx():
    c=SimpleNamespace()
    def save(*args):c.saved_tensors=args
    c.save_for_backward=save;return c
# Keep the actual contiguous decorator while bypassing only AMP bookkeeping.
fwd=m.ChunkRetentionFunction.forward.__wrapped__;bwd=m.ChunkRetentionFunction.backward.__wrapped__
def inputs(B=1,H=2,T=65,K=17,V=33):return Tensor((B,H,T,K)),Tensor((B,H,T,K)),Tensor((B,H,T,V))
def success(name,B=1,H=2,T=65,K=17,V=33,seed=True,final=True,incoming=True,terminal=True,copies=False):
    events.clear();allocations.clear();launches.clear()
    q,k,v=inputs(B,H,T,K,V);initial=Tensor((B,H,K,V)) if seed else None
    if copies:
        for t in (q,k,v,initial):
            if t is not None:t.contig=False
    c=ctx();o,ht=fwd(c,q,k,v,initial,final)
    assert current==17 and o.shape==v.shape and (ht is not None)==final
    assert c.saved_tensors[:3]==tuple(next((e[2] for e in events if e[0]=='copy' and e[1] is t),t) for t in (q,k,v))
    assert c.saved_tensors[3].shape==(B,H,((T+63)//64)*K,V)
    if copies:
        assert all(t.token is u.token for t,u in zip((q,k,v),c.saved_tensors[:3]))
    forward_names=[x[0] for x in launches]
    assert forward_names==(['fwd_kernel_h']+(['fwd_kernel_o'] if T else []) if B*H else [])
    grad=Tensor(v.shape) if incoming else None;term=Tensor((B,H,K,V)) if terminal and final else None
    if copies:
        for t in (grad,term):
            if t is not None:t.contig=False
    result=bwd(c,grad,term)
    assert current==17 and len(result)==5 and result[3:]==(None,None)
    assert result[0].shape==q.shape and result[1].shape==k.shape and result[2].shape==v.shape
    sums=[e for e in events if e[0]=='sum'];assert len(sums)==1 and result[2] is sums[0][2]
    assert [x[0] for x in launches[len(forward_names):]]==(['bwd_kernel_dh','bwd_kernel_dqkv'] if B*H and T else [])
    if grad is None:assert any(e[0]=='zeros' for e in events)
    records.append(name)
for t in (0,1,63,64,65,129):success('tail-and-empty',T=t)
for k,v in ((1,1),(15,17),(16,32),(17,33),(33,65),(64,64),(65,67),(129,7)):success('tile-rules',K=k,V=v,copies=True)
for seed in (False,True):
    for final in (False,True):
        for incoming in (False,True):
            for terminal in (False,True):success('option-branches',seed=seed,final=final,incoming=incoming,terminal=terminal)
for b,h in ((0,2),(2,0),(0,0),(2,3)):success('batch-head-branches',B=b,H=h)
def reject(name,change):
    q,k,v=inputs();args=[q,k,v,None,False];change(args);launches.clear()
    try:fwd(ctx(),*args)
    except (TypeError,ValueError,OverflowError):pass
    else:raise AssertionError(name)
    assert not launches;records.append(name)
for field,value in (('dtype','fp16'),('is_cuda',False),('layout','sparse'),('neg',True),('conj',True),('pointer',1),('nbytes',1),('pointer',2**64-4)):
    reject(field,lambda a,f=field,v=value:setattr(a[0],f,v))
reject('rank',lambda a:setattr(a[0],'ndim',3))
reject('shape',lambda a:setattr(a[1],'shape',(1,2,64,17)))
reject('flag',lambda a:a.__setitem__(4,1))
reject('seed',lambda a:a.__setitem__(3,Tensor((1,2,16,33))))
for dims in ((1,1,1,0,1),(1,1,1,1,0),(1,65536,1,1,1),(1,1,2**31,1,1),(1,1,1,2**31,1)):
    try:m._plan(*dims)
    except (ValueError,OverflowError):pass
    else:raise AssertionError(dims)
    records.append('plan-bound')
nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name.startswith('chunk_retention_') and 'kernel' in n.name]
assert hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()=='c15b51a9ae755f2f355087fc862d9ae5b35e85832161333d84f29a8e674544ff'
original=HERE.parents[3]/'bench/tritonbench_g/chunk_retention/chunk_retention.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='72133dd77de29c7b46dfa4be51e54f53b9effa8e9984155aa404fff7d3fddb86'
records.append('original-kernel-pins')
print(json.dumps(dict(cases=len(records),actual_host=True,original_copies=True,all_options=True,current_device_restored=True,gpu_execution=False)))
