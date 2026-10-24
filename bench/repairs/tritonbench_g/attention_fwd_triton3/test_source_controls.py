"""CPU interpreter controls for all three outputs; not GPU conformance."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import torch,importlib.util,json,math,hashlib
D=Path(__file__).resolve().parent;source=D/'attention_fwd_triton3_repaired.py'
spec=importlib.util.spec_from_file_location('checked_attention',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def tensor(shape,strides,fill=0):
 n=17+sum((max(s-1,0))*t for s,t in zip(shape,strides))+1+19
 base=torch.full((n,),fill,dtype=torch.float32)
 return base.as_strided(shape,strides,17),base

def check(name,NQ=3,NK=5,BM=64,BN=64,Z=1,H=2,HK=1,DIM=16,window=None,comp=False,init=True,end=True):
 R=(NQ+63)//64*64;written=(NQ+BM-1)//BM*BM
 q,_=tensor((Z,H,NQ,DIM),(H*(NQ+3)*(DIM*2+3),(NQ+3)*(DIM*2+3),DIM*2+3,2))
 k,_=tensor((Z,HK,NK,DIM),(HK*(NK+2)*(DIM+3),(NK+2)*(DIM+3),DIM+3,1))
 v,_=tensor((Z,HK,NK,DIM),(HK*(NK+5)*(DIM*3+1),(NK+5)*(DIM*3+1),DIM*3+1,3))
 o,ob=tensor((Z,H,R,DIM),(H*(R+4)*(DIM*2+5),(R+4)*(DIM*2+5),DIM*2+5,2),999.)
 for z in range(Z):
  for h in range(HK):
   for j in range(NK):v[z,h,j]=z*20+h*3+j/8+torch.arange(DIM)/16
 o.fill_(3.);m=torch.full((Z,H,R),1.);l=torch.full((Z,H,R),2.)
 obefore=ob.clone();beforeM=m.clone();beforeL=l.clone();expectedO=o.double().clone();expectedM=m.double().clone();expectedL=l.double().clone()
 for z in range(Z):
  for h in range(H):
   for r in range(written):
    keys=[j for j in range(NK) if window is None or ((r-j+window[0]>=window[1]) if comp else (0<=r-j+window[0]<window[1]))]
    center=0. if init and keys else (-math.inf if init else 1.)
    den=(0. if init and keys else 1. if init else 2.)+len(keys)*(1. if init else .5)
    numerator=torch.full((DIM,),0. if init else 3.,dtype=torch.float64)
    if keys:numerator+=v[z,h//(H//HK),keys].double().sum(0)*(1. if init else .5)
    expectedO[z,h,r]=numerator/den if end else numerator
    expectedM[z,h,r]=center+math.log2(den) if end else center
    if not end:expectedL[z,h,r]=den
 offset,size=(None,None) if window is None else window
 mod._attn_fwd[((NQ+BM-1)//BM,Z*H)](q,k,v,.25,m,o,l,*q.stride(),*k.stride(),*v.stride(),*o.stride(),Z,H,HK,NQ,R,NK,offset,size,BLOCK_M=BM,BLOCK_N=BN,BLOCK_DMODEL=DIM,END=end,INIT=init,SLIDING_WINDOW=window is not None,COMPLEMENT_SLIDING_WINDOW=comp,num_warps=4,num_stages=4)
 torch.testing.assert_close(o.double(),expectedO,rtol=2e-6,atol=2e-6)
 torch.testing.assert_close(m.double(),expectedM,rtol=2e-6,atol=2e-6)
 torch.testing.assert_close(l.double(),expectedL,rtol=0,atol=0)
 allowed=torch.zeros_like(ob,dtype=torch.bool)
 for z in range(Z):
  for h in range(H):
   for r in range(written):
    for d in range(DIM):allowed[o.storage_offset()+z*o.stride(0)+h*o.stride(1)+r*o.stride(2)+d*o.stride(3)]=True
 assert torch.equal(ob[~allowed],obefore[~allowed])
 return dict(name=name,NQ=NQ,NK=NK,block_m=BM,block_n=BN,Z=Z,H=H,HK=HK,D=DIM,window=window,complement=comp,init=init,end=end,all_OML=True,frame=True,status='PASS')
cases=[]
for init in (False,True):
 for end in (False,True):
  for window,comp,label in [(None,False,'none'),((0,3),False,'window'),((0,3),True,'older')]:
   cases.append(check(f'{label}-init{init}-end{end}',window=window,comp=comp,init=init,end=end))
for bm in (16,32,64):
 cases.append(check(f'ragged-gqa-b{bm}',65,70,bm,bm,2,6,2,16,(-2,7),False,True,False))
 cases.append(check(f'window-overflow-b{bm}',1,2,bm,bm,1,1,1,16,(2**31-1,2**31-1),False,True,True))
 cases.append(check(f'negative-offset-b{bm}',3,5,bm,bm,1,2,1,16,(-2**31,2**31-1),True,True,True))
for nk,window,label in [(0,None,'empty-kv'),(5,(-100,3),'no-loop'),(5,(0,0),'masked'),(5,(0,-2),'negative-size')]:
 cases.append(check(label,NK=nk,window=window))
print(json.dumps(dict(cases=len(cases),all_OML=True,gpu_execution=False)))
