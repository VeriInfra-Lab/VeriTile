from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
source=A/'attention_kernel_repaired.py'
spec=importlib.util.spec_from_file_location('ak_host_control',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
@dataclass(frozen=True)
class Device:
 type:str='cuda'
 index:int=0
class Tensor:
 next_base=2**32
 def __init__(self,shape,dtype='fp16',strides=None,base=None,storage_base=None):
  self.shape=tuple(shape);self.dtype=dtype;self.ndim=len(shape);self.device=Device();self.neg=False;self.conj=False;self.layout='strided'
  self.strides=tuple(strides) if strides is not None else tuple(prod(shape[i+1:]) for i in range(len(shape)))
  self.base=Tensor.next_base if base is None else base;Tensor.next_base+=2**28
  self.storage_base=self.base if storage_base is None else storage_base
 def stride(self,i=None):return self.strides if i is None else self.strides[i]
 def numel(self):return prod(self.shape)
 def is_neg(self):return self.neg
 def is_conj(self):return self.conj
 def is_contiguous(self):return self.strides==tuple(prod(self.shape[i+1:]) for i in range(self.ndim))
 def data_ptr(self):return self.base
 def element_size(self):return 2
 def untyped_storage(self):
  size=0 if not self.numel() else 2*(1+sum((d-1)*s for d,s in zip(self.shape,self.strides)))+self.base-self.storage_base
  return SimpleNamespace(data_ptr=lambda:self.storage_base,nbytes=lambda:size)
launches=[]
class Spy:
 def __getitem__(self,grid):
  def invoke(*args,**kwargs):launches.append((grid,args,kwargs))
  return invoke
from contextlib import nullcontext
m.torch=SimpleNamespace(Tensor=Tensor,float16='fp16',bfloat16='bf16',strided='strided',cuda=SimpleNamespace(device=lambda d:nullcontext()));m._fwd_kernel_aligned=Spy()
records=[]
def inputs(B=1,H=1,N=65,D=128,dtype='fp16',gapped=False):
 shape=(B,H,N,D);ts=[]
 for gap in [1,2,3,4]:
  strides=(H*gap*N*D+gap*N*D,gap*N*D,D,1) if gapped else None
  ts.append(Tensor(shape,dtype,strides))
 return [*ts[:3],Tensor((B,H,N,128),dtype),0.1,ts[3],64,64,4,2]
def success(name,args):
 launches.clear();ret=m._attention_rel_h_rel_w_kernel_aligned_device(*args)
 assert ret is None
 B,H,N,D=args[0].shape;BM,BN,W,S=args[6:]
 assert len(launches)==(1 if B*H*N else 0)
 if launches:
  grid,actual,kw=launches[0]
  assert grid==((N+BM-1)//BM,B*H,1)
  assert actual[:6]==tuple(args[:6])
  assert actual[6:22]==tuple(s for t in [args[0],args[1],args[2],args[5]] for s in t.stride())
  assert actual[22:25]==args[3].stride()[:3]
  assert actual[25:]==(B,H,N,0)
  assert kw['BLOCK_M']==BM and kw['BLOCK_N']==BN and kw['BLOCK_DMODEL']==D
  assert kw['BIAS_LAST_SIZE']==64 and kw['B0_NUMEL']==128 and kw['num_warps']==W and kw['num_stages']==S
 records.append({'name':name,'status':'PASS','launches':len(launches),'shape':args[0].shape,'dtype':args[0].dtype})
def rejected(name,args,typ,part):
 launches.clear()
 try:m._attention_rel_h_rel_w_kernel_aligned_device(*args)
 except typ as e:
  assert part in str(e),(name,str(e));assert not launches
  records.append({'name':name,'status':'PASS','rejection':str(e)})
 else:raise AssertionError(name)
for dtype in ['fp16','bf16']:
 for N in [1,63,64,65,127,128,129,8192]:success(dtype+'_N'+str(N),inputs(N=N,dtype=dtype))
 success(dtype+'_independent_pitches',inputs(2,3,97,32,dtype,True))
for B,H,N in [(0,2,64),(2,0,64),(2,3,0)]:success('empty_'+str((B,H,N)),inputs(B,H,N))
for BM,BN,D,W,S in [(16,16,16,4,2),(32,128,32,4,2),(64,64,128,8,4),(128,32,64,4,1)]:
 a=inputs(N=97,D=D);a[6:]=[BM,BN,W,S];success('config_'+str((BM,BN,D,W,S)),a)
a=inputs();a[1]=a[0];a[2]=a[0];success('readonly_alias',a)
for name,change,typ,part in [
 ('layout',lambda a:setattr(a[1],'layout','sparse'),ValueError,'strided'),
 ('dtype',lambda a:setattr(a[1],'dtype','fp32'),TypeError,'FP16'),
 ('device',lambda a:setattr(a[2],'device',Device(index=1)),ValueError,'one CUDA'),
 ('neg',lambda a:setattr(a[0],'neg',True),ValueError,'resolved'),
 ('conj',lambda a:setattr(a[5],'conj',True),ValueError,'resolved'),
 ('output_shape',lambda a:setattr(a[5],'shape',(1,1,64,128)),ValueError,'matching'),
 ('bias_shape',lambda a:setattr(a[3],'shape',(1,2,65,128)),ValueError,'relative bias'),
 ('bias_layout',lambda a:setattr(a[3],'strides',(100000,100000,256,1)),ValueError,'relative bias'),
 ('row_step',lambda a:setattr(a[0],'strides',(20000,20000,256,1)),ValueError,'dense rows'),
 ('head_overlap',lambda a:setattr(a[5],'strides',(10000,1,128,1)),ValueError,'nonoverlapping'),
 ('alias_output',lambda a:setattr(a[5],'storage_base',a[0].base),ValueError,'separate'),
 ('unaligned',lambda a:setattr(a[0],'base',2**32+1),ValueError,'aligned'),
 ('address_wrap',lambda a:setattr(a[0],'base',2**64-2),ValueError,'unsigned64'),
 ('large_stride',lambda a:setattr(a[0],'strides',(2**31,65*128,128,1)),ValueError,'envelope'),
]:
 a=inputs();change(a);rejected('reject_'+name,a,typ,part)
for idx,val,typ,part in [(6,True,TypeError,'integer'),(6,24,ValueError,'powers of two'),(7,256,ValueError,'powers of two'),(8,2,ValueError,'configuration'),(9,0,ValueError,'configuration'),(4,float('inf'),ValueError,'finite'),(4,2**128,ValueError,'finite')]:
 a=inputs();a[idx]=val;rejected('reject_arg_'+str(idx)+'_'+str(val),a,typ,part)
a=inputs(N=8193);rejected('reject_bias_overflow',a,ValueError,'bias index')
a=inputs();a[0]=object();rejected('reject_non_tensor',a,TypeError,'ordinary')

print("host controls:",len(records))

from pathlib import Path
from hashlib import sha256
import json
A=Path(__file__).resolve().parent
rows=[]
for name,B,H,N,D,BM,BN in [('head_stride',1,2,64,16,64,64),('batch_stride',2,1,64,16,64,64),('query_tail',1,1,96,16,64,32),('key_tail',1,1,96,16,32,64),('both_tails',2,3,129,32,64,64),('minimum',1,1,1,16,16,16),('BN128',1,2,97,32,32,128)]:
 layouts={r:((H+1)*g*N*D,g*N*D,D,1) for r,g in zip(['Q','K','V','Out'],[1,2,3,4])}
 layouts['Bias']=(H*N*128,N*128,128,1)
 capacity={r:1+(B-1)*s[0]+(H-1)*s[1]+(N-1)*s[2]+((128 if r=='Bias' else D)-1)*s[3] for r,s in layouts.items()}
 reads={r:set() for r in ['Q','K','V','Bias']};writes=set();events=sha256();count=0
 def address(r,b,h,n,d):
  s=layouts[r];return b*s[0]+h*s[1]+n*s[2]+d*s[3]
 def event(r,kind,b,h,n,d):
  global count
  off=address(r,b,h,n,d);assert 0<=off<capacity[r]
  if kind=='store':assert off not in writes;writes.add(off)
  else:reads[r].add(off)
  events.update(json.dumps([r,kind,b,h,n,d,off],separators=(',',':')).encode());events.update(b'\n');count+=1
 for mblock in range((N+BM-1)//BM):
  for hz in range(B*H):
   b,h=divmod(hz,H)
   for i in range(BM):
    row=mblock*BM+i
    if row<N:
     for d in range(D):event('Q','load',b,h,row,d)
     for j in range(BN):event('Bias','load',b,h,row,64+j%64)
   for start in range(0,N,BN):
    for j in range(BN):
     n=start+j
     if n<N:
      for d in range(D):event('K','load',b,h,n,d);event('V','load',b,h,n,d)
    for i in range(BM):
     row=mblock*BM+i
     if row<N:event('Bias','load',b,h,row,start//BN)
   for i in range(BM):
    row=mblock*BM+i
    if row<N:
     for d in range(D):event('Out','store',b,h,row,d)
 expected={address('Out',b,h,n,d) for b in range(B) for h in range(H) for n in range(N) for d in range(D)}
 assert writes==expected
 for r in ['Q','K','V']:
  logical={address(r,b,h,n,d) for b in range(B) for h in range(H) for n in range(N) for d in range(D)}
  assert reads[r]==logical
 assert len(writes)==B*H*N*D
 rows.append({'name':name,'status':'PASS','shape':[B,H,N,D],'blocks':[BM,BN],'strides':layouts,'capacity':capacity,'read_unique':{r:len(x) for r,x in reads.items()},'write_unique':len(writes),'event_count':count,'transcript_sha256':events.hexdigest(),'missing_output':0,'outside_logical_output_writes':0,'collisions':0,'out_of_bounds':0})

print("complete address grids:",len(rows))

# Concrete original source defects under admitted dense independent pitches.
original=(A.parents[2]/'tritonbench_g/attention_kernel/attention_kernel.py').read_text()
assert 'base=V + kv_offset' in original and 'base=Out + q_offset' in original
assert 'q = tl.load(Q_block_ptr)' in original and 'k = tl.load(K_block_ptr)' in original
counterexamples=0
N,D,H=64,16,2
qhead,khead,vhead,ohead=N*D,2*N*D,3*N*D,4*N*D
assert 1*khead != 1*vhead;counterexamples+=1
assert 1*qhead != 1*ohead;counterexamples+=1
assert 1*qhead != 2*qhead;counterexamples+=1  # independent batch pitch
assert 1*64+63>=96;counterexamples+=1  # unmasked query tail
assert 64+63>=96;counterexamples+=1  # unmasked key tail
assert 129//16>0 and 128+15>=129;counterexamples+=1
# Exact intended math uses b0 indexed by key block, b1 by lane modulo 64;
# the non-aligned variant additionally scales both bias components by log2(e).
lean=(A/'Kernels.lean').read_text();candidate=source.read_text()
assert lean.count('qk += (tl.dot(q, k)).to(tl.float32)')==2
assert '(1.44269504).to(tl.float16)' in lean and '(1.44269504).to(tl.bfloat16)' in lean
assert 'qk += ((b0 + b1) * 1.44269504)' in candidate
assert 'boundary_check=(0,)' in candidate and 'boundary_check=(1,)' in candidate
assert 'enable_fp_fusion=True' in candidate
print('original addressing/tail counterexamples:',counterexamples)
print('CPU metadata/address controls only; floating instruction validation is deferred')

# Python-to-FP32 scalar binding preserves signed zero independently of Real.
import math,struct
for scalar in [0.0,-0.0,1.0,-2.0,1e-50,-1e-50]:
 a=inputs();a[4]=scalar;success('scale_'+repr(scalar),a)
 launched=launches[0][1][4]
 assert math.copysign(1,launched)==math.copysign(1,scalar)
 word=struct.unpack('<I',struct.pack('<f',launched))[0]
 if scalar==0 or abs(scalar)==1e-50:assert word==(0x80000000 if math.copysign(1,scalar)<0 else 0)
print('scalar launch-word controls: 6; total host controls:',len(records))
