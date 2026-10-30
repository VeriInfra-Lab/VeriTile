"""Execute all three retained kernels with complete storage-frame controls.

CPU interpretation is diagnostic source evidence, not native GPU conformance.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('triton_attention_source',HERE/'Whole001/triton_attention_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(20261019)
records=[]
def tensor(shape,fill=991.,gap=1):
    count=math.prod(shape)
    backing=torch.full((17+gap*count+23,),fill,dtype=torch.float32)
    strides=tuple(gap*math.prod(shape[i+1:]) for i in range(len(shape)))
    return backing.as_strided(shape,strides,17),backing
def frame(backing,before,view=None):
    allowed=torch.zeros_like(backing,dtype=torch.bool)
    if view is not None:allowed.as_strided(view.shape,view.stride(),view.storage_offset()).fill_(True)
    allowed=allowed.repeat_interleave(4)
    assert torch.equal(backing.view(torch.uint8)[~allowed],before[~allowed])
def run(B,H,N,D,scale=.125,gradient_gap=1,alias=False,singleton=False):
    shape=(B,H,N,D);rowshape=(B*H,N)
    q,qb=tensor(shape);k,kb=tensor(shape);v,vb=tensor(shape)
    for x in (q,k,v):x.copy_(torch.randn_like(x)*.1)
    if alias:v,vb=q,qb
    gradient,gb=tensor(shape,gap=gradient_gap);gradient.copy_(torch.randn_like(gradient)*.1)
    if singleton:gradient=gradient.as_strided(shape,(7,13,D,1));assert gradient.contiguous() is gradient
    o,ob=tensor(shape);l,lb=tensor(rowshape);maximum,mb=tensor(rowshape)
    dq,dqb=tensor(shape);dk,dkb=tensor(shape);dv,dvb=tensor(shape)
    newdo,nb=tensor(shape);delta,db=tensor(rowshape)
    arrays=(qb,kb,vb,gb,ob,lb,mb,dqb,dkb,dvb,nb,db)
    initial=[x.view(torch.uint8).clone() for x in arrays]
    grid=(N//128,B*H,1)
    m._fwd_kernel[grid](q,k,v,scale,l,maximum,o,*q.stride(),*k.stride(),*v.stride(),*o.stride(),B,H,N,B*H*N,
        BLOCK_M=128,BLOCK_N=128,BLOCK_DMODEL=D,num_warps=4 if D<=64 else 8,num_stages=2)
    forward=[x.view(torch.uint8).clone() for x in (ob,lb,mb)]
    copied=gradient.contiguous()  # original source operation, including identity branch
    assert torch.equal(copied.view(torch.uint8),gradient.contiguous().view(torch.uint8))
    dq.zero_()  # original zeros_like allocation occurs before preprocess
    m._bwd_preprocess[(grid[0]*grid[1],)](o,copied,l,newdo,delta,BLOCK_M=128,D_HEAD=D)
    m._bwd_kernel[(grid[1],)](q,k,v,scale,o,newdo,dq,dk,dv,l,maximum,delta,*q.stride(),*k.stride(),*v.stride(),
        B,H,N,B*H*N,grid[0],BLOCK_M=128,BLOCK_N=128,BLOCK_DMODEL=D,num_warps=8,num_stages=1)
    # Independent dense causal attention and its autograd VJP.
    rq=q.detach().clone().requires_grad_(True);rk=k.detach().clone().requires_grad_(True);rv=v.detach().clone().requires_grad_(True)
    scores=(rq@rk.transpose(-1,-2))*float(scale)
    scores=scores.masked_fill(torch.arange(N)[None,:]>torch.arange(N)[:,None],-torch.inf)
    weights=torch.softmax(scores,-1);reference=weights@rv
    gradients=torch.autograd.grad((reference*gradient).sum(),(rq,rk,rv))
    torch.testing.assert_close(o,reference,rtol=.025,atol=.0004)
    for actual,expected in zip((dq,dk,dv),gradients):torch.testing.assert_close(actual,expected,rtol=.06,atol=.0006)
    expected_m=scores.detach().amax(-1)
    expected_l=torch.exp(scores.detach()-expected_m[...,None]).sum(-1)
    torch.testing.assert_close(maximum.view(B,H,N),expected_m,rtol=.001,atol=.00002)
    torch.testing.assert_close(l.view(B,H,N),expected_l,rtol=.001,atol=.0001)
    torch.testing.assert_close(newdo,gradient/l.view(B,H,N,1),rtol=.001,atol=.00001)
    torch.testing.assert_close(delta.view(B,H,N),(o*(gradient/l.view(B,H,N,1))).sum(-1),rtol=.001,atol=.00001)
    for backing,before in zip(arrays[:4],initial[:4]):frame(backing,before)
    for backing,before in zip((ob,lb,mb),forward):frame(backing,before)
    for backing,before,view in zip(arrays[4:],initial[4:],(o,l,maximum,dq,dk,dv,newdo,delta)):frame(backing,before,view)
    records.append((B,H,N,D,scale,gradient_gap,alias,singleton))
for d in (16,32,64,128):
    run(1,2,128,d)
    run(1,1,256,d,gradient_gap=2)
run(2,3,384,16,scale=1,gradient_gap=2)
run(1,2,256,32,scale=.75,alias=True)
run(1,1,128,16,singleton=True)
print(json.dumps(dict(cases=len(records),all_outputs_and_scratch=True,full_byte_frames=True,original_gradient_copy=True,gpu_execution=False)))
