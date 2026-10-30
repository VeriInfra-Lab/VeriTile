"""Host guards and complete forward/backward CPU interpretation; no GPU run."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
from contextlib import nullcontext
from unittest.mock import patch
import ast,importlib.util,os,sys
os.environ['TRITON_INTERPRET']='1';os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
import torch,triton
torch.set_num_threads(1)
A=Path(__file__).resolve().parent;source=A/'parallel_retention_attention_repaired.py'
spec=importlib.util.spec_from_file_location('retention_contract',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
real_torch=m.torch;kernels=[m.parallel_retention_fwd_kernel,m.parallel_retention_bwd_kernel]
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp32',device=Device()):
  self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device;self.layout='strided';self.neg=self.conj=False;self.dense=True;self.ptr_reads=0
  self.base=Tensor.next_base;Tensor.next_base+=((prod(shape)*4+15)//16)*16+2**20
 def numel(self):return prod(self.shape)
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def data_ptr(self):self.ptr_reads+=1;return self.base
 def contiguous(self):
  if self.dense:return self
  t=Tensor(self.shape,self.dtype,self.device);copies.append((self,t));return t
 def sum(self,dim):assert dim==0;return empty(*self.shape[1:],dtype=self.dtype,device=self.device)
 def to(self,dtype):assert dtype==self.dtype;return self
allocations=[];copies=[];calls=[];devices=[]
def empty(*shape,dtype,device):
 if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
 t=Tensor(shape,dtype,device);allocations.append(t);return t
class Spy:
 def __init__(self,name):self.name=name
 def __getitem__(self,grid):
  def launch(*args,**kw):calls.append((self.name,grid,args,kw))
  return launch
class Context:
 def save_for_backward(self,*args):self.saved_tensors=args
m.torch=SimpleNamespace(Tensor=Tensor,float32='fp32',strided='strided',empty=empty,cuda=SimpleNamespace(device=lambda d:(devices.append(d) or nullcontext())))
m.parallel_retention_fwd_kernel,m.parallel_retention_bwd_kernel=Spy('forward'),Spy('backward')
def inputs(B=1,H=2,T=65,DK=17,DV=33):return [Tensor((B,H,T,DK)),Tensor((B,H,T,DK)),Tensor((B,H,T,DV))]
admissions=rejections=0
def admit(a,copy=False):
 global admissions
 allocations.clear();copies.clear();calls.clear();devices.clear()
 if copy:
  for x in a:x.dense=False
 m._check_inputs(*a);ctx=Context();out=m.ParallelRetentionFunction.forward(ctx,*a);assert out.shape==a[2].shape
 saved=ctx.saved_tensors;assert len(copies)==(3 if copy else 0)
 assert all(s is (copies[i][1] if copy else a[i]) for i,s in enumerate(saved))
 incoming=Tensor(out.shape);incoming.dense=False;g=m.ParallelRetentionFunction.backward(ctx,incoming)
 assert [x.shape for x in g]==[a[0].shape,a[1].shape,a[2].shape]
 active=prod(a[0].shape[:3])>0;assert len(calls)==2*active and len(devices)==2*active
 if active:
  assert calls[0][2][:3]==saved and calls[1][2][:3]==saved
  assert calls[1][2][3] is copies[-1][1]
 else:assert all(t.ptr_reads==0 for t in a+list(saved))
 admissions+=1
for T in [1,31,32,63,64,65,128,129]:
 for DK,DV in [(1,3),(17,33),(128,129),(129,257)]:admit(inputs(T=T,DK=DK,DV=DV),T%2==1)
for dims in [(0,2,65),(1,0,65),(1,2,0)]:
 a=inputs(*dims)
 for t in a:t.base=1
 admit(a)
a=inputs();a[1]=a[0];admit(a)
a=inputs();a[0].base+=4;admit(a)
def reject(a,part):
 global rejections
 calls.clear()
 try:m._check_inputs(*a);m.ParallelRetentionFunction.forward(Context(),*a)
 except (TypeError,ValueError) as ex:assert part in str(ex),(part,str(ex));assert not calls
 else:raise AssertionError('accepted '+part)
 rejections+=1
for edit,part in [(lambda a:a.__setitem__(0,object()),'ordinary'),(lambda a:setattr(a[1],'dtype','fp16'),'FP32'),(lambda a:setattr(a[0],'device',Device('cpu')),'CUDA'),(lambda a:setattr(a[1],'device',Device(index=1)),'share'),(lambda a:setattr(a[2],'shape',(1,2,66,33)),'expected'),(lambda a:setattr(a[2],'layout','sparse'),'strided'),(lambda a:setattr(a[0],'neg',True),'flags'),(lambda a:setattr(a[1],'conj',True),'flags'),(lambda a:setattr(a[0],'base',a[0].base+1),'aligned'),(lambda a:setattr(a[2],'base',2**64-4),'address room')]:
 a=inputs();edit(a);reject(a,part)
for a,part in [(inputs(DK=0),'positive'),(inputs(DV=0),'positive'),(inputs(T=2**31),'coordinates'),(inputs(DK=2**31),'coordinates'),(inputs(T=65536*64),'axes'),(inputs(B=65536,H=1),'axes'),(inputs(DK=2**30,DV=2**30),'axes')]:reject(a,part)
for attr,value in [('neg',True),('dtype','fp16'),('shape',(1,1,1,32)),('device',Device('cpu'))]:
 a=inputs();ctx=Context();m.ParallelRetentionFunction.forward(ctx,*a);g=Tensor(a[2].shape);setattr(g,attr,value);calls.clear()
 try:m.ParallelRetentionFunction.backward(ctx,g)
 except ValueError:assert not calls;rejections+=1
 else:raise AssertionError('invalid cotangent accepted')
print('host controls:',admissions,'admissions,',rejections,'rejections')
m.torch=real_torch;m.parallel_retention_fwd_kernel,m.parallel_retention_bwd_kernel=kernels
# Mock only CUDA metadata admission/context; real tensor copies, autograd and all
# source kernel bodies execute. Independent double precision formula is the oracle.
original_check=m._check_tensor
def cpu_check(x,name):
 assert type(x) is torch.Tensor and x.layout==torch.strided and x.dtype==torch.float32 and not x.is_neg() and not x.is_conj()
m._check_tensor=cpu_check
torch.manual_seed(17);cases=0
with patch.object(torch.cuda,'device',lambda d:nullcontext()):
 for B,H,T,DK,DV in [(1,1,1,1,1),(1,2,31,3,17),(1,2,65,17,33),(1,1,129,32,64),(1,1,33,129,129),(2,3,17,3,5)]:
  shapes=[(B,H,T,DK),(B,H,T,DK),(B,H,T,DV)]
  vals=[(torch.randint(-2,3,(*shape[:-1],shape[-1]*2)).float()/16)[...,::2].requires_grad_() for shape in shapes]
  old=[x.detach().clone() for x in vals];q,k,v=[x.detach().double().requires_grad_() for x in vals]
  t=torch.arange(T);distance=t[:,None]-t[None,:];gamma=1-torch.exp2(-5-torch.arange(H,dtype=torch.float64));decay=gamma[:,None,None]**distance.clamp_min(0);decay=decay*(distance>=0)
  ref=((q@k.transpose(-2,-1))*decay[None,:,:,:])@v*(DK**-0.5)
  out=m.parallel_retention(*vals);torch.testing.assert_close(out,ref.detach().float(),rtol=8e-5,atol=2e-6)
  g=torch.ones((),dtype=torch.float32).expand_as(out) if cases%2==0 else torch.randn_like(out)/16
  expected=torch.autograd.grad(ref,(q,k,v),g.double());out.backward(g)
  for x,want,before in zip(vals,expected,old):
   torch.testing.assert_close(x.grad,want.float(),rtol=2e-4,atol=3e-6)
   assert torch.equal(x.detach().view(torch.int32),before.view(torch.int32))
  cases+=1
print('complete source forward/backward, noncontiguous input copies, broadcast cotangents and input byte frames:',cases)
original=A.parents[2]/'tritonbench_g/parallel_retention_attention/parallel_retention_attention.py';tree=ast.parse(original.read_text());text=original.read_text()
assert 'torch.randn(1, n_heads, seq_len, d_head' in text or 'q = torch.randn' in text
# The original minimum tile is 1 for DK=1, invalid for the supported tl.dot
# compiler shapes. The repaired min16 choice and explicit zero padding are bound.
assert 'BK = min(128, triton.next_power_of_2(k.shape[-1]))' in text
assert 'padding_option="zero"' not in text
assert m._checked_geometry(1,1,1,1,1)
print('original missing minimum feature tile and undefined padded block-load controls: 2')
print('CPU interpreter only; no GPU numerical conformance claimed')
