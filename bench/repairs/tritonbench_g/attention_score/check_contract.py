"""Direct-input host controls and source interpreter checks; no CUDA execution."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
from contextlib import nullcontext
import ast,importlib.util,itertools,os,sys,types,warnings
os.environ['TRITON_INTERPRET']='1';os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
import torch,triton
real_torch,real_triton=torch,triton;torch.set_num_threads(1)
A=Path(__file__).resolve().parent;source=A/'attention_score_repaired.py'
spec=importlib.util.spec_from_file_location('score_contract',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);kernel=m._score_kernel
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp32',device=Device()):
  self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device;self.neg=self.conj=False;self.layout='strided';self.contiguous=True
  self.base=Tensor.next_base;Tensor.next_base+=2**30;self.zeroed=False
 @property
 def is_cuda(self):return self.device.type=='cuda'
 def is_contiguous(self):return self.contiguous
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def size(self,i):return self.shape[i]
 def numel(self):return prod(self.shape)
 def data_ptr(self):return self.base
allocations=[]
def zeros(shape,dtype,device):
 out=Tensor(shape,dtype,device);out.zeroed=True;allocations.append(out);return out
class ResourceError(Exception):pass
class Spy:
 def __init__(self):self.failures=0;self.calls=[]
 def __getitem__(self,grid):
  def launch(*args,**kw):
   self.calls.append((grid(kw),args,kw))
   if self.failures:self.failures-=1;raise ResourceError('simulated prelaunch resource failure')
  return launch
spy=Spy();m._score_kernel=spy
m.torch=SimpleNamespace(Tensor=Tensor,float32='fp32',strided='strided',zeros=zeros,cuda=SimpleNamespace(device=lambda d:nullcontext()))
m.triton=SimpleNamespace(cdiv=triton.cdiv,OutOfResources=ResourceError)
def inputs(Z=1,H=4,HK=2,N=65,NK=64,D=16,R=None):
 if R is None:R=N
 return [Tensor((Z,H,N,D)),Tensor((Z,HK,NK,D)),Tensor((Z,H,R)),None,False]
admissions=rejections=0
def admit(a,block=64,retry=False):
 global admissions
 m._BLOCK_M=m._BLOCK_N=block;spy.calls.clear();spy.failures=int(retry);allocations.clear()
 with warnings.catch_warnings():warnings.simplefilter('ignore');out=m.get_score(*a)
 q,k,normalizer,window,complement=a;Z,H,N,D=q.shape;HK,NK=k.shape[1:3]
 assert out is allocations[0] and out.shape==(Z,H,NK) and out.dtype=='fp32' and out.device==q.device and out.zeroed
 live=bool(N*out.numel());assert len(spy.calls)==(1+int(retry) if live else 0)
 selected=max(16,block//2) if live and retry else block
 assert m._BLOCK_M==m._BLOCK_N==selected
 if live:
  grid,args,kw=spy.calls[-1]
  assert grid==((NK+selected-1)//selected,Z*H)
  assert args[0] is q and args[1] is k and args[2] is normalizer and args[4] is out
  assert kw['BLOCK_M']==kw['BLOCK_N']==selected and kw['enable_fp_fusion']
  assert args[5:16]==(H*N*D,N*D,D,1,HK*NK*D,NK*D,D,1,H*NK,NK,1)
 admissions+=1;return out

def reject(a,part):
 global rejections
 m._BLOCK_M=m._BLOCK_N=64;spy.calls.clear();allocations.clear()
 try:m.get_score(*a)
 except (ValueError,TypeError) as e:assert part in str(e),(part,str(e));assert not spy.calls
 else:raise AssertionError('accepted '+part)
 rejections+=1
for block in [16,32,64]:
 for N,NK in [(1,1),(63,64),(64,63),(65,64),(65,65),(129,97)]:
  for window,comp in [(None,False),((0,32),False),((-5,17),True)]:
   a=inputs(N=N,NK=NK,R=N+7);a[3:]=[window,comp];admit(a,block)
 admit(inputs(),block,True)
for a in [inputs(N=0,NK=65),inputs(NK=0),inputs(Z=0),inputs(H=0,HK=1)]:
 a[0].base+=1;admit(a)
a=inputs(H=2,HK=2,N=65,NK=65);a[1]=a[0];admit(a)
for edit,part in [
 (lambda a:setattr(a[0],'dtype','fp16'),'FP32'),
 (lambda a:setattr(a[0],'device',Device('cpu')),'CUDA'),
 (lambda a:setattr(a[1],'device',Device(index=1)),'one CUDA'),
 (lambda a:setattr(a[1],'shape',(1,3,64,16)),'divide'),
 (lambda a:setattr(a[1],'shape',(1,0,64,16)),'positive'),
 (lambda a:setattr(a[2],'shape',(1,4,64)),'every query'),
 (lambda a:setattr(a[0],'shape',(1,4,65,24)),'matching'),
 (lambda a:setattr(a[0],'contiguous',False),'contiguous'),
 (lambda a:setattr(a[2],'layout','sparse'),'strided'),
 (lambda a:setattr(a[0],'neg',True),'resolved'),
 (lambda a:setattr(a[1],'conj',True),'resolved'),
 (lambda a:setattr(a[0],'base',a[0].base+1),'aligned'),
 (lambda a:setattr(a[2],'base',2**64-4),'unsigned'),
 (lambda a:a.__setitem__(3,(2**31,1)),'signed int32'),
 (lambda a:a.__setitem__(3,(1,1.5)),'integers'),
 (lambda a:a.__setitem__(4,1),'bool'),
]:
 a=inputs();edit(a);reject(a,part)
a=inputs(D=8);reject(a,'power of two')
a=inputs(Z=65536,H=1,HK=1,N=1,NK=1);reject(a,'grid')
a=inputs(N=2**31-1,NK=1,R=2**31-1);reject(a,'signed int32')
print('host controls:',admissions,'admissions,',rejections,'rejections')
# Complete current-source interpreter executions with a countable independent formula.
m.torch=real_torch;m.triton=real_triton;m._score_kernel=kernel

def launch(fn,q,k,norm,B,window=None,comp=False):
 Z,H,N,D=q.shape;HK,NK=k.shape[1:3];R=norm.shape[-1];out=torch.full((Z,H,NK),-17,dtype=torch.float32)
 off,size=window or (0,0)
 fn[(triton.cdiv(NK,B),Z*H)](q,k,norm,1/(D**0.5),out,*q.stride(),*k.stride(),*out.stride(),Z,H,HK,N,R,NK,off,size,SLIDING_WINDOW=window is not None,COMPLEMENT_SLIDING_WINDOW=comp,BLOCK_M=B,BLOCK_N=B,BLOCK_DMODEL=D,enable_fp_fusion=True)
 return out
cases=0
for B,N,NK in [(16,1,1),(16,17,16),(32,31,33),(32,33,32),(64,63,64),(64,65,64),(64,65,65),(64,129,97)]:
 q=torch.zeros(2,4,N,16);k=torch.zeros(2,2,NK,16);norm=torch.zeros(2,4,N+7);before=[t.clone() for t in [q,k,norm]]
 for win,comp in [(None,False),((0,9),False),((-3,7),True),((2**31-1,2**31-1),True)]:
  out=launch(kernel,q,k,norm,B,win,comp)
  i=torch.arange(N,dtype=torch.int64)[:,None];j=torch.arange(NK,dtype=torch.int64)[None,:]
  if win is None:allowed=torch.ones(N,NK,dtype=torch.bool)
  else:
   dist=i-j+win[0];allowed=dist>=win[1] if comp else (dist>=0)&(dist<win[1])
  ref=allowed.sum(0).float().expand(2,4,NK)
  assert torch.equal(out,ref),(B,N,NK,win,comp,out,ref)
  assert all(torch.equal(t,old) for t,old in zip([q,k,norm],before));cases+=1
print('complete FP32 interpreter grids/input frames:',cases)
# Original query-tail bug: even key length leaves the padded query contribution live.
original=A.parents[2]/'tritonbench_g/attention_score/attention_score.py'
tree=ast.parse(original.read_text(),str(original));tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test_') or isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id.startswith('_BLOCK') for t in n.targets)]
orig=types.ModuleType('original_score');orig.__file__=str(original);sys.modules[orig.__name__]=orig;exec(compile(tree,str(original),'exec'),orig.__dict__)
q=torch.zeros(1,1,65,16);k=torch.zeros(1,1,64,16);norm=torch.zeros(1,1,128)
assert torch.all(launch(orig._score_kernel,q,k,norm,64)==128)
assert torch.all(launch(kernel,q,k,norm,64)==65)
q=torch.zeros(1,1,2,16);k=torch.zeros(1,1,1,16);norm=torch.zeros(1,1,2)
assert torch.all(launch(kernel,q,k,norm,16,(2**31-1,2**31-1),True)==2)
# Signed-i32 source distance wraps at the second query row in the original.
assert (torch.tensor(1,dtype=torch.int32)+2147483647).item()==-2147483648
x=torch.ones(1,1,1,16);negative=torch._neg_view(x)
assert negative.is_contiguous() and negative.data_ptr()==x.data_ptr() and not torch.equal(x,negative)
print('original query-tail, signed-distance, and logical-negative counterexamples: 3')
print('CPU metadata/interpreter only; no GPU arithmetic claim')
