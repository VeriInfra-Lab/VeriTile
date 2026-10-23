"""Actual source in Triton's CPU interpreter. No native/GPU conformance claim."""
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('attn_direct',HERE/'attn_fwd_triton_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(20261019)
records=[]
def view(shape,stride_kind,fill=0,dtype=torch.float32):
    b,h,n,d=shape
    stride=(h*(n+3)*(d*stride_kind+5),(n+3)*(d*stride_kind+5),d*stride_kind+5,stride_kind)
    size=17+sum((a-1)*s for a,s in zip(shape,stride))+1+23
    storage=torch.full((size,),fill,dtype=dtype)
    return storage.as_strided(shape,stride,17),storage

def run(N,D,B=1,H=2,alias=False,uniform=False,sign=1):
    bd=max(16,1<<(D-1).bit_length())
    q,qb=view((B,H,N,D),2);k,kb=view((B,H,N,D),1)
    v,vb=(k,kb) if alias else view((B,H,N,D),3)
    o,ob=view((B,H,N,D),2,999,torch.bfloat16)
    q.copy_(torch.zeros_like(q) if uniform else torch.randn_like(q)*.1)
    k.copy_(torch.ones_like(k) if uniform else torch.randn_like(k)*.1)
    if not alias:v.copy_(torch.ones_like(v) if uniform else torch.randn_like(v)*.1)
    qs=torch.full((B*H*(math.ceil(N/128)),),float(sign))
    ks=torch.linspace(.5,1.5,B*H*math.ceil(N/64))
    before=[x.clone() for x in (qb,kb,vb,ob,qs,ks)]
    m._attn_fwd_repaired[(math.ceil(N/128),B*H,1)](q,k,v,qs,ks,o,*q.stride(),*k.stride(),*v.stride(),*o.stride(),B,H,N_CTX=N,HEAD_DIM=D,BLOCK_DMODEL=bd,BLOCK_M=128,BLOCK_N=64,STAGE=3,num_warps=8,num_stages=3)
    scores=q@k.transpose(-1,-2)
    for b in range(B):
        for h in range(H):
            scores[b,h]*=qs[(b*H+h)*math.ceil(N/128)+torch.arange(N)//128,None]
            scores[b,h]*=ks[(b*H+h)*math.ceil(N/64)+torch.arange(N)//64][None,:]
    scores.masked_fill_(torch.arange(N)[None,:]>torch.arange(N)[:,None],-torch.inf)
    reference=torch.softmax(scores*math.log(2),-1)@v
    torch.testing.assert_close(o.float(),reference,rtol=.025,atol=.005)
    for old,new in zip(before,(qb,kb,vb,ob,qs,ks)):
        if new is not ob:assert torch.equal(old,new)
    allowed=torch.zeros_like(ob,dtype=torch.bool)
    allowed.as_strided(o.shape,o.stride(),o.storage_offset()).fill_(True)
    assert torch.equal(ob[~allowed],before[3][~allowed])
    records.append(dict(N=N,D=D,B=B,H=H,alias=alias,scale_sign=sign,output=True,frame=True,status='PASS'))
for n,d in [(1,1),(1,16),(3,31),(63,96),(64,128),(65,129),(127,16),(128,31),(129,96)]:
    run(n,d)
    run(n,d,alias=True,uniform=True,sign=0)
for sign in (-1,0,1):run(129,16,B=2,H=3,sign=sign)
print(json.dumps(dict(cases=len(records),all_output_cells=True,full_frames=True,gpu_execution=False)))
