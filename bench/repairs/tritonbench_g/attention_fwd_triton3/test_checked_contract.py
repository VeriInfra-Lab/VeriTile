"""Actual host guard and persistent retry controls; no GPU execution."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from math import prod
import ast, importlib.util, json, sys, warnings
sys.dont_write_bytecode=True
source=Path(__file__).with_name('attention_fwd_triton3_repaired.py')
spec=importlib.util.spec_from_file_location('aft3_host',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Tensor:
    next_base=2**40
    def __init__(self,shape,strides=None):
        self.shape=tuple(shape);self.ndim=len(shape)
        self.strides=tuple(strides) if strides else tuple(prod(shape[i+1:]) for i in range(len(shape)))
        self.layout='strided';self.dtype='fp32';self.device='cuda:0';self.is_cuda=True
        self.neg=False;self.conj=False;self.contig=True
        self.base=Tensor.next_base;Tensor.next_base+=2**28
        self.available=mod._span(self.shape,self.strides)
    def stride(self,i=None):return self.strides if i is None else self.strides[i]
    def is_contiguous(self):return self.contig
    def is_neg(self):return self.neg
    def is_conj(self):return self.conj
    def data_ptr(self):return self.base
    def untyped_storage(self):return SimpleNamespace(nbytes=lambda:4*self.available)
    def storage_offset(self):return 0
    def __getitem__(self,key):
        shape=[len(range(*s.indices(n))) for n,s in zip(self.shape,key)]
        return Tensor(shape)
    def contiguous(self):return self
    def to(self,dtype):assert dtype==self.dtype;return self
class Resources(Exception):pass
class Spy:
    def __init__(self,failures=()):self.calls=[];self.failures=list(failures)
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            self.calls.append((grid(kwargs),args,kwargs))
            if self.failures:
                error=self.failures.pop(0)
                if error:raise error
        return launch
mod.torch=SimpleNamespace(Tensor=Tensor,strided='strided',float32='fp32',cuda=SimpleNamespace(device=lambda _:nullcontext()))
mod.triton=SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d,OutOfResources=Resources)
def inputs(nq=3,nk=5,h=2,hk=1,d=16,z=1):
    r=(nq+63)//64*64
    return [Tensor((z,h,nq,d)),Tensor((z,hk,nk,d)),Tensor((z,hk,nk,d)),
            Tensor((z,h,r,d)),Tensor((z,h,r)),Tensor((z,h,r))]
def invoke(a,**kw):return mod._forward(*a[:3],.25,*a[3:],**kw)
rows=[]
def reject(name,a,**kw):
    mod._BLOCK_M=mod._BLOCK_N=64;spy=Spy();mod._attn_fwd=spy
    try:invoke(a,**kw)
    except (TypeError,ValueError):assert not spy.calls;rows.append(name)
    else:raise AssertionError(name)
for dim in (16,32,64,128):
    for init in (False,True):
        for end in (False,True):
            a=inputs(d=dim);mod._BLOCK_M=mod._BLOCK_N=64;spy=Spy();mod._attn_fwd=spy
            out,m,l=invoke(a,init=init,end=end,sliding_window=(0,3))
            assert m is a[4] and l is a[5]
            assert out is a[3] if not end else out.shape==(1,2,3,dim)
            assert spy.calls[0][0]==(1,2)
            assert spy.calls[0][2]['INIT']==init and spy.calls[0][2]['END']==end
            rows.append('flags_dim')
for nq,nk in [(0,5),(3,0),(65,70),(1,2)]:
    a=inputs(nq,nk);mod._attn_fwd=Spy()
    invoke(a,init=True,end=True,sliding_window=(2**31-1,2**31-1))
    rows.append('tail_empty_extreme_window')
for field,value in [('layout','sparse'),('dtype','fp16'),('is_cuda',False),('device','cuda:1'),('neg',True),('conj',True),('base',1),('base',2**64-4),('available',0)]:
    a=inputs();setattr(a[1],field,value);reject(field,a)
a=inputs();a[0]=None;reject('nontensor',a)
a=inputs();a[3].shape=(1,2,3,16);reject('unpadded_O',a)
a=inputs();a[4].contig=False;reject('strided_M',a)
a=inputs();a[3].strides=(2048,1024,1,1);reject('overlapping_O',a)
a=inputs();a[1].shape=(1,3,5,16);a[2].shape=a[1].shape;reject('invalid_GQA',a)
a=inputs();a[1].base=a[0].base;reject('alias',a)
reject('nonbool',inputs(),init=1)
reject('window_int64',inputs(),sliding_window=(2**31,3))
reject('initial_window_pointer_overflow',inputs(),sliding_window=(2**31-1,0))
reject('window_shape',inputs(),sliding_window=(1,))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mod._BLOCK_M=mod._BLOCK_N=64
    for before,after in [(64,32),(32,16)]:
        a=inputs();spy=Spy((Resources('capacity'),None));mod._attn_fwd=spy
        out,m,l=invoke(a)
        assert [x[2]['BLOCK_M'] for x in spy.calls]==[before,after]
        assert mod._BLOCK_M==mod._BLOCK_N==after and out is a[3]
        rows.append('persistent_retry')
    mod._BLOCK_M=mod._BLOCK_N=64;spy=Spy((Resources('first'),Resources('second')));mod._attn_fwd=spy
    try:invoke(inputs())
    except Resources:assert mod._BLOCK_M==mod._BLOCK_N==32;rows.append('second_failure_keeps_globals')
    else:raise AssertionError('second failure swallowed')
    mod._BLOCK_M=mod._BLOCK_N=64;spy=Spy((RuntimeError('other'),));mod._attn_fwd=spy
    try:invoke(inputs())
    except RuntimeError:assert mod._BLOCK_M==mod._BLOCK_N==64;rows.append('other_failure_no_retry')
    else:raise AssertionError('other failure swallowed')
# The wrapper still uses only the original intentional return materialization.
tree=ast.parse(source.read_text())
forward=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_forward')
assert ast.unparse(forward.body[-1].value)=='(o, m, l)'
assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='contiguous' for n in ast.walk(forward))==1
print(json.dumps({'status':'PASS','controls':len(rows),'GPU_execution':False}))
