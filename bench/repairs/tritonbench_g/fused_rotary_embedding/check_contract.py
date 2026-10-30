"""Host guard controls and full actual-kernel CPU interpreter; no GPU run."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
from contextlib import nullcontext
import ast,importlib.util,os,sys,types
os.environ['TRITON_INTERPRET']='1';os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
import torch,triton
torch.set_num_threads(1)
A=Path(__file__).resolve().parent;source=A/'fused_rotary_embedding_repaired.py'
spec=importlib.util.spec_from_file_location('rotary_contract',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
kernel=m.decoding_fused_rotary_embedding_kernel;real_torch=m.torch;copies=[]
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp32',device=Device(),data=None):
  self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=device;self.layout='strided';self.neg=self.conj=False;self.contiguous=True;self.offset=0;self.data=data
  self.base=(Tensor.next_base+15)//16*16;self.storage_base=self.base;self.storage_bytes=max(1,self.numel())*4;Tensor.next_base=self.base+self.storage_bytes+2**20
  stride=[];n=1
  for size in reversed(self.shape):stride.insert(0,n);n*=max(1,size)
  self.strides=tuple(stride)
 @property
 def is_cuda(self):return self.device.type=='cuda'
 def numel(self):return prod(self.shape)
 def stride(self,i=None):return self.strides if i is None else self.strides[i]
 def is_contiguous(self):return self.contiguous
 def storage_offset(self):return self.offset
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def data_ptr(self):return self.base
 def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.storage_base,nbytes=lambda:self.storage_bytes)
 def to(self,device):
  assert device=='cpu';copies.append(self);return SimpleNamespace(tolist=lambda:self.data)
class Spy:
 def __init__(self):self.calls=[]
 def __getitem__(self,grid):
  def launch(*args,**kw):self.calls.append((grid,args,kw))
  return launch
spy=Spy();m.decoding_fused_rotary_embedding_kernel=spy;m.torch=SimpleNamespace(Tensor=Tensor,float32='fp32',int32='int32',strided='strided',cuda=SimpleNamespace(device=lambda d:nullcontext()))
def inputs(T=3,HQ=6,HK=2,D=16,B=4,X=None):
 new=X is not None;X=X or D
 return [Tensor((T,HQ,D)),Tensor((T,HK,D)),Tensor((T,HK,D)),Tensor((T+1,D//2+1)),Tensor((T+2,D//2+3)),Tensor((8,HK,D//X,B,X) if new else (8,HK,B,D)),Tensor((8,HK,B,D)),Tensor((T,4),'int32',data=[[t//B,-7,-8,-9] for t in range(T)]),Tensor((T,),'int32',data=[t%B+1 for t in range(T)]),new]
admissions=rejections=0
def admit(a):
 global admissions
 spy.calls.clear();copies.clear();assert m.decoding_fused_rotary_embedding(*a) is None
 T,HQ,D=a[0].shape;assert len(spy.calls)==int(T>0)
 assert copies==([a[8],a[7]] if T else [])
 if T:
  grid,args,kw=spy.calls[0];assert grid==(HQ,T) and all(x is y for x,y in zip(args[:9],a[:9]))
  assert kw['HEAD_DIM']==D and kw['KV_GROUP_NUM']==HQ//a[1].shape[1] and kw['enable_fp_fusion']
  assert kw['num_warps']==(16 if D>=512 else 8 if D>=256 else 4)
 admissions+=1

def reject(a,part):
 global rejections
 spy.calls.clear();copies.clear()
 try:m.decoding_fused_rotary_embedding(*a)
 except (ValueError,TypeError) as e:assert part in str(e),(part,str(e));assert not spy.calls
 else:raise AssertionError('accepted '+part)
 rejections+=1
for T,HQ,HK in [(1,1,1),(3,6,2),(7,8,4)]:
 for D in [2,6,16,96,128,256,512]:
  admit(inputs(T,HQ,HK,D));admit(inputs(T,HQ,HK,D,X=2))
# No pointers or metadata copies on empty paths; unused strides remain arbitrary.
a=inputs(T=0)
for t in a[:3]+a[7:9]:t.strides=(1,)*t.ndim;t.base=1;t.storage_base=0
admit(a)
for edit,part in [
 (lambda a:a.__setitem__(0,object()),'q must'),
 (lambda a:setattr(a[1],'dtype','fp16'),'FP32'),
 (lambda a:setattr(a[2],'device',Device('cpu')),'FP32'),
 (lambda a:setattr(a[3],'device',Device(index=1)),'FP32'),
 (lambda a:setattr(a[4],'layout','sparse'),'FP32'),
 (lambda a:setattr(a[0],'offset',1),'zero-offset'),
 (lambda a:setattr(a[2],'neg',True),'zero-offset'),
 (lambda a:setattr(a[3],'conj',True),'zero-offset'),
 (lambda a:setattr(a[4],'contiguous',False),'contiguous'),
 (lambda a:setattr(a[2],'strides',(1,1,1)),'strides'),
 (lambda a:setattr(a[8],'dtype','int64'),'int32'),
 (lambda a:setattr(a[0],'base',a[0].base+1),'aligned'),
 (lambda a:setattr(a[1],'storage_base',a[1].storage_base+4),'allocation starts'),
 (lambda a:setattr(a[2],'storage_bytes',4),'storage extent'),
 (lambda a:setattr(a[7],'data',[[-1]*4]*3),'selected block'),
 (lambda a:setattr(a[8],'data',[0,1,2]),'length must'),
 (lambda a:setattr(a[8],'data',[17,1,2]),'length must'),
 (lambda a:setattr(a[8],'data',[1,1,1]),'distinct'),
 (lambda a:a.__setitem__(9,1),'must be bool'),
]:
 a=inputs();edit(a);reject(a,part)
a=inputs();a[2]=a[1];reject(a,'independent')
a=inputs();a[2].base=a[1].base+4;a[2].storage_base=a[2].base;reject(a,'independent')
a=inputs();a[0].base=a[0].storage_base=2**64-4*a[0].numel();reject(a,'formed kernel pointers')
a=inputs();a[7]=Tensor((2,4),'int32');a[8]=Tensor((2,),'int32');reject(a,'per token')
reject(inputs(D=15),'positive even');reject(inputs(HQ=3,HK=2),'multiple')
reject(inputs(T=65536),'grid axis');reject(inputs(T=1,HQ=2**31,HK=1,D=2),'grid axis')
print('host controls:',admissions,'admissions,',rejections,'rejections')
m.torch=real_torch;m.decoding_fused_rotary_embedding_kernel=kernel

def launch(fn,a,original=False):
 q,k,v,c,s,kc,vc,table,length,new=a;T,HQ,D=q.shape;HK=k.shape[1];X=kc.shape[-1] if new else D;B=vc.shape[2]
 common=[q,k,v,c,s,kc,vc,table,length,X]
 if original:common += [*q.stride()[:2],*k.stride()[:2],q.stride(2),*c.stride()]
 else:common += [*q.stride(),*k.stride(),*v.stride(),*c.stride(),*s.stride(),length.stride(0)]
 common += [kc.stride(0),kc.stride(1),*(kc.stride()[-3:] if new else (0,kc.stride(2),kc.stride(3))),*vc.stride(),*table.stride(),B]
 kw={'KV_GROUP_NUM':HQ//HK,'HEAD_DIM':D,'num_warps':16 if D>=512 else 8 if D>=256 else 4,'enable_fp_fusion':True}
 if not original:kw.update(BLOCK_HALF=triton.next_power_of_2(D//2),BLOCK_DIM=triton.next_power_of_2(D))
 fn[(HQ,T)](*common,**kw)

def real_inputs(T,HQ,HK,D,B,new,X):
 def data(shape):return ((torch.arange(prod(shape))%9-4).float()/4).view(shape)
 q=data((T,HQ,D));k=data((T,HK,D));v=data((T,HK,D))
 # V includes non-arithmetic FP32 payloads and must retain their exact bits.
 if v.numel()>=4:v.view(torch.int32).flatten()[:4]=torch.tensor([0x7fc01234,-2147483648,0x7f800000,1],dtype=torch.int32)
 c=data((T+1,D//2+1));s=data((T+2,D//2+3));kc=torch.full((8,HK,D//X,B,X) if new else (8,HK,B,D),-37.0);vc=torch.full((8,HK,B,D),-39.0)
 table=torch.tensor([[t//B,-7,-8,-9] for t in range(T)],dtype=torch.int32);length=torch.tensor([t%B+1 for t in range(T)],dtype=torch.int32)
 return [q,k,v,c,s,kc,vc,table,length,new]

def target(a):
 q,k,v,c,s,kc,vc,table,length,new=a;T,HQ,D=q.shape;HK=k.shape[1];B=vc.shape[2];half=D//2
 def rotate(x):return torch.cat([x[...,:half]*c[:T,None,:half]-x[...,half:]*s[:T,None,:half],x[...,:half]*s[:T,None,:half]+x[...,half:]*c[:T,None,:half]],-1)
 qo=rotate(q);ko=rotate(k);kco=kc.clone();vco=vc.clone()
 for t in range(T):
  col,slot=divmod(int(length[t])-1,B);block=int(table[t,col]);vco[block,:,slot,:]=v[t]
  if new:
   X=kc.shape[-1]
   for d in range(D):kco[block,:,d//X,slot,d%X]=ko[t,:,d]
  else:kco[block,:,slot,:]=ko[t]
 return qo,kco,vco
cases=0
for T,HQ,HK,D,B,new,X in [(1,1,1,2,1,False,2),(3,6,2,6,3,True,3),(3,6,2,16,4,False,16),(7,8,4,96,3,True,8),(2,8,2,128,16,False,128),(2,8,2,128,16,True,8),(2,8,2,256,16,True,16),(2,8,2,512,16,False,512),(16,8,4,64,4,False,64)]:
 a=real_inputs(T,HQ,HK,D,B,new,X);old=[t.clone() for t in a[:9]];want=target(a);launch(kernel,a)
 for actual,expected in zip([a[0],a[5],a[6]],want):assert torch.equal(actual.view(torch.int32),expected.view(torch.int32)),(D,new)
 for j in [1,2,3,4,7,8]:assert torch.equal(a[j].view(torch.int32),old[j].view(torch.int32))
 cases+=1
print('complete actual-source interpreter grids, input/cache-hole byte frames:',cases)
original=A.parents[2]/'tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py'
tree=ast.parse(original.read_text(),str(original));tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
orig=types.ModuleType('original_rotary');orig.__file__=str(original);sys.modules[orig.__name__]=orig;exec(compile(tree,str(original),'exec'),orig.__dict__)
a=real_inputs(3,6,2,16,4,False,16);want=target(a);launch(orig.decoding_fused_rotary_embedding_kernel,a,True);assert not torch.equal(a[0],want[0])
# In the original fixture T=16 but metadata contains only two rows.
assert 15>=2
# Original int32 token-stride multiplication can overflow before pointer addition.
assert ((65535*65536+2**31)%2**32)-2**31<0
print('original independent-Sin-pitch failure, invalid fixture, and int32-address controls: 3')
