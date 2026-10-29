"""Host admission, original AST identity, and complete autograd CPU interpreter calls."""
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
A=Path(__file__).resolve().parent;source=A/'lightning_attention_repaired.py'
spec=importlib.util.spec_from_file_location('lightning_contract',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
original=A.parents[2]/'tritonbench_g/lightning_attention/lightning_attention.py'
original_ast={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
current_ast={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)}
for name in ['_fwd_kernel','_bwd_intra_kernel','_bwd_inter_kernel']:assert current_ast[name]==original_ast[name]
print('all three original kernel ASTs unchanged')
real_torch=m.torch;kernels=[m._fwd_kernel,m._bwd_intra_kernel,m._bwd_inter_kernel]
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp32',device=Device()):
  self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device;self.layout='strided';self.neg=self.conj=False;self.dense=True
  self.base=Tensor.next_base;Tensor.next_base+=((prod(shape)*4+15)//16)*16+2**32
 @property
 def is_cuda(self):return self.device.type=='cuda'
 def is_contiguous(self):return self.dense
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def data_ptr(self):return self.base
 def contiguous(self):
  if self.dense:return self
  t=Tensor(self.shape,self.dtype,self.device);copies.append((self,t));return t
allocations=[];copies=[];calls=[]
def empty(shape,dtype,device):
 t=Tensor(shape,dtype,device);allocations.append(t);return t
def empty_like(q):return empty(q.shape,q.dtype,q.device)
class Spy:
 def __init__(self,name):self.name=name
 def __getitem__(self,grid):
  def launch(*args,**kw):calls.append((self.name,grid,args,kw))
  return launch
class Context:
 def save_for_backward(self,*args):self.saved_tensors=args
m.torch=SimpleNamespace(Tensor=Tensor,float32='fp32',strided='strided',empty=empty,empty_like=empty_like,cuda=SimpleNamespace(device=lambda d:nullcontext()))
m._fwd_kernel,m._bwd_intra_kernel,m._bwd_inter_kernel=[Spy(s) for s in ['forward','intra','inter']]
def inputs(B=1,H=1,N=65,D=16,E=32):return [Tensor((B,H,N,D)),Tensor((B,H,N,D)),Tensor((B,H,N,E))]
admissions=rejections=0
def admit(a,broadcast=False):
 global admissions
 allocations.clear();copies.clear();calls.clear();ctx=Context();out=m.LightningAttention2NoDecay.forward(ctx,*a)
 assert out is allocations[0] and out.shape==a[2].shape and ctx.saved_tensors==tuple(a) and not copies
 assert calls[0][0]=='forward' and all(x is y for x,y in zip(calls[0][2][:4],[*a,out]))
 incoming=Tensor(out.shape);incoming.dense=not broadcast
 grads=m.LightningAttention2NoDecay.backward(ctx,incoming)
 assert grads==tuple(allocations[1:]) and [x[0] for x in calls]==['forward','intra','inter']
 assert all(call[3]['enable_fp_fusion'] for call in calls)
 assert len(copies)==int(broadcast)
 assert all(calls[1][2][i] is a[i] for i in range(3))
 assert calls[2][2][4:7]==grads and calls[1][2][4:7]==grads
 admissions+=1
for N in [1,63,64,65,127,128,129]:
 for D,E in [(16,16),(16,32),(32,64),(64,128)]:admit(inputs(N=N,D=D,E=E),N%2==1)
admit(inputs(2,8,128,64,128),True)
a=inputs(D=32,E=32);a[1]=a[0];admit(a)
a=inputs();a[0].base+=4;admit(a)
def reject(a,part):
 global rejections
 calls.clear()
 try:m.LightningAttention2NoDecay.forward(Context(),*a)
 except (TypeError,ValueError) as ex:assert part in str(ex),(part,str(ex));assert not calls
 else:raise AssertionError('accepted '+part)
 rejections+=1
for edit,part in [
 (lambda a:a.__setitem__(0,object()),'ordinary'),
 (lambda a:setattr(a[0],'dtype','fp16'),'FP32'),
 (lambda a:setattr(a[0],'device',Device('cpu')),'CUDA'),
 (lambda a:setattr(a[1],'device',Device(index=1)),'same-device'),
 (lambda a:setattr(a[1],'shape',(1,1,66,16)),'expected'),
 (lambda a:setattr(a[2],'layout','sparse'),'strided'),
 (lambda a:setattr(a[0],'dense',False),'contiguous'),
 (lambda a:setattr(a[2],'neg',True),'resolved'),
 (lambda a:setattr(a[1],'conj',True),'resolved'),
 (lambda a:setattr(a[0],'base',a[0].base+1),'aligned'),
 (lambda a:setattr(a[2],'base',2**63-4),'formation'),
]:
 a=inputs();edit(a);reject(a,part)
for a,part in [(inputs(N=0),'positive'),(inputs(D=24),'power-of-two'),(inputs(E=8),'power-of-two'),(inputs(N=65535*64+1),'axis'),(inputs(B=2**26,N=1),'offsets')]:reject(a,part)
for attr,value in [('neg',True),('dtype','fp16'),('shape',(1,1,1,32)),('device',Device('cpu'))]:
 a=inputs();ctx=Context();m.LightningAttention2NoDecay.forward(ctx,*a);g=Tensor(a[2].shape);setattr(g,attr,value);calls.clear()
 try:m.LightningAttention2NoDecay.backward(ctx,g)
 except ValueError:assert not calls;rejections+=1
 else:raise AssertionError('invalid cotangent accepted')
print('host controls:',admissions,'admissions,',rejections,'rejections')
m.torch=real_torch;m._fwd_kernel,m._bwd_intra_kernel,m._bwd_inter_kernel=kernels
# Actual torch autograd executes the repaired wrapper with CPU interpreter kernels;
# CUDA admission/context are mocked while real storage, strides, values and autograd remain active.
torch.manual_seed(13);cases=0
with patch.object(torch.Tensor,'is_cuda',property(lambda self:True)),patch.object(torch.cuda,'device',lambda d:nullcontext()):
 for B,H,N,D,E in [(1,1,1,16,16),(1,1,63,16,32),(1,1,64,16,16),(1,1,65,32,64),(1,1,129,16,32),(2,3,65,16,32),(2,8,128,64,128)]:
  values=[(torch.randint(-2,3,shape).float()/8).requires_grad_() for shape in [(B,H,N,D),(B,H,N,D),(B,H,N,E)]]
  old=[x.detach().clone() for x in values];q,k,v=[x.detach().double().requires_grad_() for x in values]
  causal=torch.ones(N,N,dtype=torch.bool).tril();ref=((q@k.transpose(-2,-1))*causal)@v
  out=m.lightning_attn2_no_decay(*values)
  assert torch.equal(out,ref.detach().float()),(N,D,E,(out-ref).abs().max())
  g=(torch.randint(-2,3,out.shape).float()/8) if cases%2 else torch.ones((),dtype=torch.float32).expand_as(out)
  expected=torch.autograd.grad(ref,(q,k,v),g.double())
  if cases%2:out.backward(g)
  else:out.sum().backward()
  for x,want,before in zip(values,expected,old):
   assert torch.equal(x.grad,want.float()),(N,D,E,(x.grad-want).abs().max())
   assert torch.equal(x.detach().view(torch.uint8),before.view(torch.uint8))
  cases+=1
print('complete forward + intra + inter autograd interpreter calls/input frames:',cases)
x=torch.ones(1,1,1,16);negative=torch._neg_view(x)
assert negative.is_contiguous() and negative.data_ptr()==x.data_ptr() and negative.is_neg()
print('lazy-negative pointer/logical-value counterexample reproduced; rejected by checked contract')
print('CPU metadata/interpreter only; no GPU numerical claim')
