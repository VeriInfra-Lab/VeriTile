"""Checked host controls and complete CPU interpreter grids; no GPU execution."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
from contextlib import nullcontext
import ast,importlib.util,os,sys,types
os.environ['TRITON_INTERPRET']='1';os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
import torch,triton
torch.set_num_threads(1)
A=Path(__file__).resolve().parent;source=A/'attn_fwd_causal_repaired.py'
spec=importlib.util.spec_from_file_location('causal_contract',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
kernel=m._attn_fwd;real_torch=m.torch
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp32',device=Device()):
  self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device;self.layout='strided';self.neg=self.conj=False
  self.base=(Tensor.next_base+15)//16*16;self.storage_base=self.base;self.storage_bytes=max(1,self.numel())*self.element_size();Tensor.next_base=self.base+self.storage_bytes+2**32
  stride=[];n=1
  for size in reversed(self.shape):stride.insert(0,n);n*=max(1,size)
  self.strides=tuple(stride)
 def numel(self):return prod(self.shape)
 def element_size(self):return 4 if self.dtype=='fp32' else 2
 def stride(self,i=None):return self.strides if i is None else self.strides[i]
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def data_ptr(self):return self.base
 def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.storage_base,nbytes=lambda:self.storage_bytes)
allocations=[]
def empty(shape,device,dtype):
 t=Tensor(shape,dtype,device);allocations.append(t);return t
def empty_like(q,dtype):return empty(q.shape,q.device,dtype)
class Spy:
 def __init__(self):self.calls=[]
 def __getitem__(self,grid):
  def launch(*args,**kw):self.calls.append((grid,args,kw))
  return launch
spy=Spy();m._attn_fwd=spy;m.torch=SimpleNamespace(Tensor=Tensor,float32='fp32',float16='fp16',strided='strided',empty=empty,empty_like=empty_like,cuda=SimpleNamespace(device=lambda d:nullcontext()))
def inputs(B=1,H=1,N=128):return [Tensor((B,H,N,128)) for _ in range(3)]+[Tensor((B,H,(N+127)//128)),Tensor((B,H,(N+63)//64))]
admissions=rejections=0
def admit(a):
 global admissions
 allocations.clear();spy.calls.clear();out=m.forward(*a);B,H,N,D=a[0].shape
 assert out is allocations[0] and out.shape==a[0].shape and out.dtype=='fp16' and out.device==a[0].device
 assert len(allocations)==2 and allocations[1].shape==(B,H,N) and allocations[1].dtype=='fp32'
 assert len(spy.calls)==int(bool(B*H*N))
 if spy.calls:
  grid,args,kw=spy.calls[0];assert grid==((N+127)//128,B*H,1)
  assert all(x is y for x,y in zip(args[:6],[*a,out]))
  assert (kw['BLOCK_M'],kw['BLOCK_N'],kw['HEAD_DIM'],kw['STAGE'],kw['num_warps'],kw['num_stages'])==(128,64,128,3,8,2)
  assert kw['enable_fp_fusion']
 admissions+=1

def reject(a,part):
 global rejections
 allocations.clear();spy.calls.clear()
 try:m.forward(*a)
 except (ValueError,TypeError) as e:assert part in str(e),(part,str(e));assert not spy.calls
 else:raise AssertionError('accepted '+part)
 rejections+=1
for B,H in [(1,1),(2,3),(1,7)]:
 for N in [1,63,64,65,127,128,129,257]:admit(inputs(B,H,N))
for B,H,N in [(0,2,129),(2,0,129),(2,3,0)]:
 a=inputs(B,H,N)
 for t in a:t.base+=1
 admit(a)
# Empty-strided tensors are admitted with their exact unused strides.
fixture=torch.empty_strided((0,1,1,128),(1,1,1,1))
assert fixture.stride()==(1,1,1,1) and fixture.numel()==0
for B,H,N in [(0,1,1),(1,0,1),(1,1,0),(0,1,5_000_000)]:
 a=inputs(B,H,N)
 for t in a[:3]:t.strides=(1,1,1,1)
 admit(a)
a=inputs(N=1)
for t in a:t.base+=4;t.storage_bytes+=4
admit(a)
for edit,part in [
 (lambda a:a.__setitem__(0,object()),'ordinary'),
 (lambda a:setattr(a[0],'dtype','fp16'),'FP32'),
 (lambda a:setattr(a[2],'device',Device('cpu')),'CUDA'),
 (lambda a:setattr(a[2],'device',Device(index=1)),'one device'),
 (lambda a:setattr(a[2],'layout','sparse'),'strided'),
 (lambda a:setattr(a[2],'neg',True),'ordinary'),
 (lambda a:setattr(a[3],'conj',True),'ordinary'),
 (lambda a:setattr(a[1],'shape',(1,1,129,128)),'matching'),
 (lambda a:setattr(a[3],'shape',(1,1,2)),'scales'),
 (lambda a:setattr(a[2],'strides',(16384,16384,1,128)),'canonical'),
 (lambda a:setattr(a[4],'strides',(3,3,1)),'canonical'),
 (lambda a:setattr(a[2],'storage_base',a[1].storage_base+4),'disjoint'),
 (lambda a:setattr(a[0],'base',a[0].base+1),'aligned'),
 (lambda a:setattr(a[4],'base',2**64-4),'nonwrapping'),
]:
 a=inputs();edit(a);reject(a,part)
a=inputs();a[1]=a[0];reject(a,'disjoint')
a=inputs();a[:3]=[Tensor((1,1,128,96)) for _ in range(3)];reject(a,'width')
reject(inputs(65536,1,1),'integer bounds');reject(inputs(1,1,2**24),'integer bounds')
print('host controls:',admissions,'admissions,',rejections,'rejections')
m.torch=real_torch;m._attn_fwd=kernel

def launch(fn,q,k,v,qs,ks,out):
 B,H,N,D=q.shape
 fn[(triton.cdiv(N,128),B*H,1)](q,k,v,qs,ks,out,*q.stride(),*k.stride(),*v.stride(),*out.stride(),B,H,N_CTX=N,HEAD_DIM=128,BLOCK_M=128,BLOCK_N=64,STAGE=3,num_warps=8,num_stages=2,enable_fp_fusion=True)
 return out
cases=0
for B,H,N in [(1,1,n) for n in [1,63,64,65,127,128,129,257]]+[(2,3,65)]:
 q=torch.zeros(B,H,N,128);k=torch.zeros_like(q)
 v=(torch.arange(128)%8).float().div(8).expand(B,H,N,128).contiguous()
 qs=torch.ones(B,H,triton.cdiv(N,128));ks=torch.ones(B,H,triton.cdiv(N,64))
 values=[q,k,v,qs,ks];old=[t.clone() for t in values]
 storage=torch.full((q.numel()+10,),-17,dtype=torch.float16);out=storage[4:4+q.numel()].view_as(q)
 launch(kernel,*values,out)
 assert torch.equal(out,v.half()),(B,H,N,(out-v.half()).abs().max())
 assert torch.all(storage[:4]==-17) and torch.all(storage[-6:]==-17)
 assert all(torch.equal(t.view(torch.uint8),b.view(torch.uint8)) for t,b in zip(values,old));cases+=1
print('complete FP32 interpreter grids/input/output-hole frames:',cases)
original=A.parents[2]/'tritonbench_g/attn_fwd_causal/attn_fwd_causal.py'
tree=ast.parse(original.read_text(),str(original));tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
orig=types.ModuleType('original_causal');orig.__file__=str(original);sys.modules[orig.__name__]=orig;exec(compile(tree,str(original),'exec'),orig.__dict__)
# Original numeric samples use its working FP16 ABI; its intended FP32 PV dot is rejected by CUDA compilation.
q=torch.zeros(1,1,128,128,dtype=torch.float16);k=torch.zeros_like(q);v=torch.ones_like(q);qs=torch.ones(1,1,1);ks=torch.ones(1,1,2)
out=torch.full_like(q,-17);launch(orig._attn_fwd,q,k,v,qs,ks,out)
assert torch.all(out[...,96:]==-17) and out[...,96:].numel()==4096
q.zero_();k.zero_();v.zero_();q[0,0,0,0]=1024;k[0,0,1,0]=2048;v[0,0,1,0]=1
out.fill_(-17);launch(orig._attn_fwd,q,k,v,qs,ks,out);assert out[0,0,0,0]==1
fixed=torch.full_like(q,-17);launch(kernel,q.float(),k.float(),v.float(),qs,ks,fixed);assert fixed[0,0,0,0]==0
# At N=1 the original diagonal visits key block 1, while KS contains only block 0.
assert list(range(0,128,64))==[0,64] and list(range(0,min(128,1),64))==[0]
print('original missing-feature, future-leak, and key-scale-tail counterexamples: 3')
print('CPU metadata/interpreter only; no GPU arithmetic claim')
