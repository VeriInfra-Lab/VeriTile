"""Host guards, address/byte frames and exact zero-QK CPU interpreter controls."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import nullcontext
from math import prod
import ast,importlib.util,itertools,os,struct,sys
os.environ['TRITON_INTERPRET']='1';os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import torch,triton
real_torch=torch;torch.set_num_threads(1)
A=Path(__file__).resolve().parent;source=A/'attention_kernel_aligned_repaired.py'
spec=importlib.util.spec_from_file_location('aligned_contract',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
kernels=[m._fwd_kernel_aligned_fp16,m._fwd_kernel_aligned_bf16]
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp16',strides=None):
  self.shape=tuple(shape);self.dtype=dtype;self.ndim=len(shape);self.device=Device();self.neg=False;self.conj=False;self.layout='strided'
  self.strides=tuple(strides) if strides is not None else tuple(prod(shape[i+1:]) for i in range(len(shape)))
  self.base=Tensor.next_base;Tensor.next_base+=2**28;self.storage_base=self.base
 def stride(self,i=None):return self.strides if i is None else self.strides[i]
 def numel(self):return prod(self.shape)
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def is_contiguous(self):
  if not self.numel():return True
  expected=1
  for extent,stride in reversed(list(zip(self.shape,self.strides))):
   if extent>1 and stride!=expected:return False
   expected*=extent
  return True
 def data_ptr(self):return self.base
 def untyped_storage(self):
  size=0 if not self.numel() else 2*(1+sum((d-1)*s for d,s in zip(self.shape,self.strides)))+self.base-self.storage_base
  return SimpleNamespace(data_ptr=lambda:self.storage_base,nbytes=lambda:size)
launches=[]
class Spy:
 def __init__(self,dtype):self.dtype=dtype
 def __getitem__(self,grid):
  def invoke(*args,**kw):launches.append((self.dtype,grid,args,kw))
  return invoke
m.torch=SimpleNamespace(Tensor=Tensor,float16='fp16',bfloat16='bf16',strided='strided',cuda=SimpleNamespace(device=lambda d:nullcontext()))
m._fwd_kernel_aligned_fp16=Spy('fp16');m._fwd_kernel_aligned_bf16=Spy('bf16')
def inputs(B=1,H=1,N=65,D=64,dtype='fp16'):
 ts=[Tensor((B,H,N,D),dtype) for _ in range(4)]
 return [*ts[:3],Tensor((B,H,N,128),dtype),0.1,ts[3],64,64,4,2]
accepted=rejected=0
def admit(args):
 global accepted
 launches.clear();result=m._attention_rel_h_rel_w_kernel_aligned_device(*args);assert result is None
 B,H,N,D=args[0].shape;BM,BN,W,S=args[6:]
 assert len(launches)==bool(B*H*N)
 if launches:
  ty,grid,actual,kw=launches[0]
  assert ty==args[0].dtype and grid==((N+BM-1)//BM,B*H,1)
  assert all(actual[i] is args[i] for i in [0,1,2,3,5])
  assert actual[6:22]==tuple(s for t in [args[0],args[1],args[2],args[5]] for s in t.stride())
  assert actual[22:25]==args[3].stride()[:3] and actual[25:]==(B,H,N,0)
  assert kw['BLOCK_M']==BM and kw['BLOCK_N']==BN and kw['BLOCK_DMODEL']==D and kw['num_warps']==W and kw['num_stages']==S and kw['enable_fp_fusion']
 accepted+=1

def reject(args,part):
 global rejected
 launches.clear()
 try:m._attention_rel_h_rel_w_kernel_aligned_device(*args)
 except (TypeError,ValueError) as e:assert part in str(e),(part,str(e));assert not launches
 else:raise AssertionError('accepted '+part)
 rejected+=1
for ty,BM,BN,D,W,S in itertools.product(['fp16','bf16'],[16,32,64,128],[16,32,64,128],[16,32,64,128],[4,8],[1,2,3,4]):
 a=inputs(N=1,D=D,dtype=ty);a[6:]=[BM,BN,W,S]
 if BN==D==128 and (S==4 or BM==128 and S==3):reject(a,'shared-memory')
 else:admit(a)
assert (accepted,rejected)==(1004,20)
for ty in ['fp16','bf16']:
 for N in [1,63,64,65,127,128,129,4096]:admit(inputs(N=N,dtype=ty))
 for B,H,N in [(0,2,65),(2,0,65),(2,2,0)]:
  a=inputs(B,H,N,dtype=ty);a[1]=a[0];a[0].base+=1;admit(a)
 a=inputs(2,3,65,32,ty)
 for i,g in zip([0,1,2,5],[1,2,3,4]):a[i].strides=(3*65*32*g*2,65*32*g*2,32*g*2,g)
 admit(a)
 a=inputs(N=1,dtype=ty);a[3].strides=(313,227,139,1);admit(a)
 for value in [0.0,-0.0,1,-2,1e-50,-1e-50]:
  a=inputs(N=1,dtype=ty);a[4]=value;admit(a)
  assert struct.pack('<f',launches[0][2][4])==struct.pack('<f',float(value))
for edit,part in [
 (lambda a:setattr(a[0],'layout','sparse'),'strided CUDA'),
 (lambda a:setattr(a[1],'device',Device('cpu')),'strided CUDA'),
 (lambda a:setattr(a[2],'device',Device(index=1)),'same-device'),
 (lambda a:setattr(a[0],'dtype','fp32'),'FP16'),
 (lambda a:setattr(a[1],'dtype','bf16'),'FP16'),
 (lambda a:setattr(a[0],'neg',True),'ordinary tensor values'),
 (lambda a:setattr(a[5],'conj',True),'ordinary tensor values'),
 (lambda a:setattr(a[5],'shape',(1,1,64,64)),'equal'),
 (lambda a:setattr(a[3],'shape',(1,1,65,64)),'bias'),
 (lambda a:setattr(a[3],'strides',(20000,20000,256,1)),'bias'),
 (lambda a:setattr(a[0],'strides',(10000,10000,64,0)),'positive'),
 (lambda a:setattr(a[5],'strides',(10000,10000,1,1)),'nonoverlapping'),
 (lambda a:setattr(a[0],'strides',(2**31,10000,64,1)),'signed-i32'),
 (lambda a:setattr(a[0],'base',a[0].base+1),'aligned'),
 (lambda a:(setattr(a[5],'base',2**64-2),setattr(a[5],'storage_base',2**64-2)),'nonwrapping'),
 (lambda a:a.__setitem__(1,a[0]),'disjoint'),
 (lambda a:a.__setitem__(5,a[0]),'disjoint'),
 (lambda a:a.__setitem__(4,float('inf')),'finite'),
 (lambda a:a.__setitem__(4,float('nan')),'finite'),
 (lambda a:a.__setitem__(6,True),'ordinary integers'),
 (lambda a:a.__setitem__(7,24),'widths'),
 (lambda a:a.__setitem__(8,2),'warps'),
]:
 a=inputs();edit(a);reject(a,part)
reject(inputs(N=4097),'64 block biases')
print('host controls',accepted,'admissions',rejected,'rejections')

# Execute the unmodified kernel body on real CPU storage via Triton's interpreter.
m.torch=real_torch;m._fwd_kernel_aligned_fp16,m._fwd_kernel_aligned_bf16=kernels

def view(B,H,N,D,mode):
 if mode=='dense':return torch.empty(B,H,N,D,dtype=torch.float16)
 if mode=='step':return torch.empty(B,H,N,D*3,dtype=torch.float16)[...,1::3]
 if mode=='permuted':return torch.empty(B,N,H,D,dtype=torch.float16).permute(0,2,1,3)
 if mode=='transpose':return torch.empty(B,H,D,N,dtype=torch.float16).transpose(2,3)
 if mode=='gap':return torch.empty(B*2,H*2,N,D,dtype=torch.float16)[::2,::2]
 raise ValueError(mode)
def launch(q,k,v,bias,o,BM,BN):
 B,H,N,D=q.shape
 kernels[0][((N+BM-1)//BM,B*H,1)](q,k,v,bias,1.0,o,*q.stride(),*k.stride(),*v.stride(),*o.stride(),*bias.stride()[:3],B,H,N,0,OUT_DTYPE=triton.language.float16,BIAS_LAST_SIZE=64,B0_NUMEL=128,BLOCK_DMODEL=D,BLOCK_M=BM,BLOCK_N=BN,num_warps=4,num_stages=2,enable_fp_fusion=True)
for N,D,BM,BN in [(1,16,16,16),(63,32,32,64),(64,64,64,64),(65,64,64,32),(97,32,32,128),(127,128,64,64),(129,16,128,16),(129,64,64,64)]:
 B,H=2,2;q,k,v,o=[view(B,H,N,D,mode) for mode in ['step','permuted','transpose','gap']]
 q.zero_();k.zero_();v.copy_((torch.arange(B*H*N*D).reshape(B,H,N,D)%17-8).float()/16)
 torch.empty(0,dtype=torch.float16).set_(o.untyped_storage(),0,(o.untyped_storage().nbytes()//2,),(1,)).fill_(-17)
 bias=torch.zeros(B,H,N,128,dtype=torch.float16);old=[bytes(t.untyped_storage()) for t in [q,k,v,bias,o]]
 launch(q,k,v,bias,o,BM,BN)
 assert torch.equal(o,v.double().mean(2,keepdim=True).expand_as(v).half()),(N,D,BM,BN)
 assert [bytes(t.untyped_storage()) for t in [q,k,v,bias]]==old[:4]
 offsets={o.storage_offset()+sum(i*s for i,s in zip(ix,o.stride())) for ix in itertools.product(range(B),range(H),range(N),range(D))}
 touched={2*x+b for x in offsets for b in [0,1]};new=bytes(o.untyped_storage())
 assert len(offsets)==o.numel() and all(b==new[i] for i,b in enumerate(old[4]) if i not in touched)
print('complete FP16 interpreter grids and byte frames: 8')
original=A.parents[2]/'tritonbench_g/attention_kernel_aligned/attention_kernel_aligned.py'
tree=ast.parse(original.read_text(),str(original));tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
import types
orig=types.ModuleType('original_aligned');orig.__file__=str(original);sys.modules[orig.__name__]=orig;exec(compile(tree,str(original),'exec'),orig.__dict__)
q=torch.zeros(1,2,64,64,dtype=torch.float16);k=q.clone();v=torch.ones_like(q);bias=torch.zeros(1,2,64,128,dtype=torch.float16)
backing=torch.full((1,4,64,64),-17,dtype=torch.float16);o=backing[:,::2]
orig._attention_rel_h_rel_w_kernel_aligned_device(q,k,v,bias,1.0,o,64,64,4,2)
assert torch.all(o[:,1]==-17) and torch.all(backing[:,1]==1)
backing.fill_(-17);launch(q,k,v,bias,o,64,64)
assert torch.all(o==1) and torch.all(backing[:,1::2]==-17)
print('original output-head counterexample reproduced and repaired')
print('CPU host/interpreter checks only; CUDA and BF16 instruction validation deferred')
