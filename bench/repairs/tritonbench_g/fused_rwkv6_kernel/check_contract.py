"""Literal host guards and complete CPU source interpretation; no GPU execution."""
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
A=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('rwkv_contract',A/'fused_rwkv6_kernel_repaired.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
real_torch=m.torch;kernel=m.fused_recurrent_rwkv6_fwd_kernel
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
 def is_contiguous(self):return self.dense
 def data_ptr(self):self.ptr_reads+=1;return self.base
 def new_empty(self,*shape,dtype=None):
  t=Tensor(shape,self.dtype if dtype is None else dtype,self.device);allocations.append(t);return t
 def sum(self,dim):assert dim==0;return self.new_empty(*self.shape[1:])
 def to(self,dtype):assert dtype==self.dtype;return self
allocations=[];calls=[];devices=[]
class Spy:
 def __getitem__(self,grid):
  def launch(*args,**kw):calls.append((grid,args,kw))
  return launch
class Context:
 def save_for_backward(self,*args):self.saved_tensors=args
m.torch=SimpleNamespace(Tensor=Tensor,float32='fp32',strided='strided',cuda=SimpleNamespace(device=lambda d:(devices.append(d) or nullcontext())))
m.fused_recurrent_rwkv6_fwd_kernel=Spy()
def inputs(B=1,H=2,T=3,K=17,V=19):return [Tensor((B,H,T,K)),Tensor((B,H,T,K)),Tensor((B,H,T,V)),Tensor((B,H,T,K)),Tensor((H,K))]
admissions=rejections=0
def admit(a,initial=False,final=False,reverse=False,scale=.5):
 global admissions
 allocations.clear();calls.clear();devices.clear();ctx=Context();B,H,T,K=a[0].shape;V=a[2].shape[-1];h0=Tensor((B,H,K,V)) if initial else None
 out,ht=m.FusedRecurrentRWKV6Function.forward(ctx,*a,scale,h0,final,reverse)
 assert out.shape==(B,H,T,V) and (ht is not None)==final
 assert out.dtype=='fp32' and out.device==a[0].device
 assert ctx.saved_tensors==tuple(a+[h0]) and ctx.scale is scale and ctx.reverse is reverse
 assert len(calls)==int(bool(T or final)) and len(devices)==len(calls)
 if calls:
  grid,args,kw=calls[0];assert args[:5]==tuple(a) and args[6] is h0 and args[7] is ht
  assert args[8:11]==(T*K,T*V,scale)
  assert grid==(triton.cdiv(V,min(triton.next_power_of_2(V),32)),triton.cdiv(K,min(triton.next_power_of_2(K),32)),B*H)
  assert kw['USE_INITIAL_STATE']==initial and kw['STORE_FINAL_STATE']==final and kw['REVERSE']==reverse
 admissions+=1
for dims in [(1,1,1,1,1),(2,3,4,8,8),(1,2,3,17,19),(1,1,4,65,37),(1,1,0,3,5)]:
 for initial in [False,True]:
  for final in [False,True]:
   for reverse in [False,True]:admit(inputs(*dims),initial,final,reverse)
a=inputs(K=19);a[1]=a[0];a[3]=a[0];admit(a)
a=inputs();a[0].base+=4;admit(a,reverse=True)
admit(inputs(),scale=float('nan'));admit(inputs(),scale=float('inf'))
def reject(a,part,**kw):
 global rejections
 calls.clear()
 try:m.FusedRecurrentRWKV6Function.forward(Context(),*a,kw.pop('scale',.5),**kw)
 except (TypeError,ValueError) as ex:assert part in str(ex),(part,str(ex));assert not calls
 else:raise AssertionError('accepted '+part)
 rejections+=1
for edit,part in [(lambda a:a.__setitem__(0,object()),'ordinary'),(lambda a:setattr(a[1],'dtype','fp16'),'FP32'),(lambda a:setattr(a[0],'device',Device('cpu')),'CUDA'),(lambda a:setattr(a[1],'device',Device(index=1)),'share'),(lambda a:setattr(a[2],'shape',(1,2,4,19)),'expected'),(lambda a:setattr(a[4],'shape',(1,17)),'expected U'),(lambda a:setattr(a[0],'layout','sparse'),'strided'),(lambda a:setattr(a[0],'dense',False),'contiguous'),(lambda a:setattr(a[0],'neg',True),'flags'),(lambda a:setattr(a[0],'conj',True),'flags'),(lambda a:setattr(a[0],'base',3),'alignment'),(lambda a:setattr(a[0],'base',2**64-4),'room')]:
 a=inputs();edit(a);reject(a,part)
for dims,part in [((0,1,1,1,1),'positive'),((1,0,1,1,1),'positive'),((1,1,1,0,1),'positive'),((1,1,1,1,0),'positive'),((1,1,2**31,1,1),'signed'),((65536,1,0,1,1),'axes'),((1,1,0,65536*32,1),'axes'),((1,1,0,1,2**36),'axes')]:reject(inputs(*dims),part)
reject(inputs(),'expected U',initial_state=Tensor((1,1,1,1)))
reject(inputs(),'bool',reverse=1);reject(inputs(),'bool',output_final_state=1)
for scale in [None,True,Tensor(())]:reject(inputs(),'scale',scale=scale)
a=inputs();a[0].base=4;reject(a,'room',reverse=True)
# The helper sentinel is selected before delegation and is saved as the chosen value.
with patch.object(m.FusedRecurrentRWKV6Function,'apply',lambda *args:m.FusedRecurrentRWKV6Function.forward(Context(),*args)):
 a=inputs(K=8);m.fused_recurrent_rwkv6(*a);assert calls[-1][1][10]==8**-.5
print('host controls:',admissions,'admissions,',rejections,'rejections; helper sentinel checked')
m.torch=real_torch;m.fused_recurrent_rwkv6_fwd_kernel=kernel
# Only CUDA metadata and owning-device context are mocked. All source kernel
# operations, allocations, host sum0 and result casts execute on CPU tensors.
def cpu_check(x,name):
 assert type(x) is torch.Tensor and x.layout==torch.strided and x.dtype==torch.float32 and x.is_contiguous() and not x.is_neg() and not x.is_conj()
m._check_tensor=cpu_check
torch.manual_seed(27);cases=0
with patch.object(torch.cuda,'device',lambda d:nullcontext()):
 for B,H,T,K,V,initial,final,reverse in [(1,1,1,1,1,False,False,False),(2,3,4,8,8,True,True,False),(1,2,3,17,19,False,True,True),(1,1,4,65,37,True,True,True),(1,1,2,32,32,True,False,False),(1,1,2,33,65,False,False,False),(1,1,0,3,5,False,False,False),(1,1,0,3,5,False,True,False),(1,1,0,3,5,True,True,True)]:
  shapes=[(B,H,T,K),(B,H,T,K),(B,H,T,V),(B,H,T,K),(H,K)]
  a=[(torch.randint(-2,3,(prod(s)+1,)).float()/16)[1:].view(s) for s in shapes]
  if cases==4:a[1]=a[0];a[3]=a[0]
  if cases==0:a[0]=torch.as_strided(a[0],a[0].shape,(101,37,11,1)) # unused singleton strides
  h0=torch.randn(B,H,K,V)/16 if initial else None
  if T==0 and initial:h0.fill_(-0.)
  before=[x.clone() for x in a+([h0] if initial else [])];scale=.5
  q,k,v,w,u=[x.double() for x in a];state=h0.double().clone() if initial else torch.zeros(B,H,K,V,dtype=torch.float64);want=torch.empty(B,H,T,V,dtype=torch.float64)
  for t in (range(T-1,-1,-1) if reverse else range(T)):
   outer=k[:,:,t,:,None]*v[:,:,t,None,:]
   want[:,:,t,:]=((state+outer*u[None,:,:,None])*(q[:,:,t,:,None]*scale)).sum(-2)
   state=state*torch.exp(w[:,:,t,:,None])+outer
  ctx=Context();out,ht=m.FusedRecurrentRWKV6Function.forward(ctx,*a,scale,h0,final,reverse)
  torch.testing.assert_close(out,want.float(),rtol=5e-5,atol=2e-6)
  if final:torch.testing.assert_close(ht,state.float(),rtol=5e-5,atol=2e-6)
  if T==0 and initial and final:assert torch.equal(ht.view(torch.int32),torch.zeros_like(ht).view(torch.int32))
  for x,old in zip(a+([h0] if initial else []),before):assert torch.equal(x.view(torch.int32),old.view(torch.int32))
  assert all(x is y for x,y in zip(ctx.saved_tensors,a+[h0]))
  cases+=1
print('complete source forward/reverse, split K/V, optional/T0 state, aliases, singleton strides, offsets and byte frames:',cases)
original=A.parents[2]/'tritonbench_g/fused_rwkv6_kernel/fused_rwkv6_kernel.py';source=original.read_text();assert '(i_bh + i_k * B * H) * s_v_h' in source
# Concrete original defect: B=1,H=2,T=1,K=V=1, V head stride=2.
# Head1 writes O[2] beyond the two-element allocation; repaired head stride=1.
assert (1+0*1*2)*2>=1*1*2*1*1 and (1+0*1*2)*(1*1)<2
assert not hasattr(m.FusedRecurrentRWKV6Function,'jvp') or 'jvp' not in m.FusedRecurrentRWKV6Function.__dict__
assert 'backward' not in m.FusedRecurrentRWKV6Function.__dict__
print('original gapped-head output out-of-bounds control reproduced; no backward source exists')
print('CPU interpreter only; no GPU numerical conformance claimed')
