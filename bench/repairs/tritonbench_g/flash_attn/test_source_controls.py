"""Complete repaired source under CPU Triton interpretation; no GPU claim."""
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('flash_source_controls',HERE/'flash_attn_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(20261019)
records=[]
def view(shape,gap,fill=0):
    b,h,n,d=shape
    strides=(h*(n+3)*(d*gap+5),(n+3)*(d*gap+5),d*gap+5,gap)
    backing=torch.full((17+sum((a-1)*s for a,s in zip(shape,strides))+1+23,),float(fill),dtype=torch.float32)
    return backing.as_strided(shape,strides,17),backing
def run(N,D,causal,scale=1,B=1,H=2):
    q,qb=view((B,H,N,D),2);k,kb=view((B,H,N,D),1);v,vb=view((B,H,N,D),3)
    o,ob=view((B,H,N,D),2,999)
    lb=torch.full((B*H*N+41,),999.,dtype=torch.float32);l=lb[17:17+B*H*N].view(B*H,N)
    q.copy_(torch.randn_like(q)*.1);k.copy_(torch.randn_like(k)*.1);v.copy_(torch.randn_like(v)*.1)
    originals=[x.view(torch.uint8).clone() for x in (qb,kb,vb,ob,lb)]
    m._fwd_kernel[((N+127)//128,B*H,1)](q,k,v,scale,l,o,*q.stride(),*k.stride(),*v.stride(),*o.stride(),B,H,N,
        BLOCK_M=128,DIM=D,BLOCK_N=64,IS_CAUSAL=causal,num_warps=4 if D<=64 else 8,num_stages=4)
    scores=(q*(float(scale)*1.44269504))@k.transpose(-1,-2)
    if causal:scores.masked_fill_(torch.arange(N)[None,:]>torch.arange(N)[:,None],-torch.inf)
    weights=torch.softmax(scores*math.log(2),-1)
    reference=weights@v
    scratch=torch.logsumexp(scores*math.log(2),-1)/math.log(2)
    torch.testing.assert_close(o,reference,rtol=.025,atol=.005)
    torch.testing.assert_close(l.view(B,H,N),scratch,rtol=.003,atol=.003)
    for old,new in zip(originals[:3],(qb,kb,vb)):assert torch.equal(old,new.view(torch.uint8))
    allowed=torch.zeros_like(ob,dtype=torch.bool)
    allowed.as_strided(o.shape,o.stride(),o.storage_offset()).fill_(True)
    byte_allowed=allowed.repeat_interleave(4)
    assert torch.equal(ob.view(torch.uint8)[~byte_allowed],originals[3][~byte_allowed])
    scratch_allowed=torch.zeros_like(lb,dtype=torch.bool);scratch_allowed[17:17+B*H*N]=True
    scratch_bytes=scratch_allowed.repeat_interleave(4)
    assert torch.equal(lb.view(torch.uint8)[~scratch_bytes],originals[4][~scratch_bytes])
    records.append((N,D,causal,scale,B,H))
for causal in (False,True):
    for d in (16,32,64,128):run(65,d,causal)
for n in (1,63,64,127,128,129):run(n,32,True)
for scale in (-1,-0.0,0,1):run(129,16,False,scale,B=2,H=3)
print(json.dumps(dict(cases=len(records),all_output_and_scratch_cells=True,full_byte_frames=True,gpu_execution=False)))
