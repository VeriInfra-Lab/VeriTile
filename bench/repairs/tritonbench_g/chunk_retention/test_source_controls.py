"""Complete retained kernels under CPU interpretation; no native GPU claim."""
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('retention_source',HERE/'Whole001/chunk_retention_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(20261019)
records=[]
def tensor(shape,fill=937.):
    count=math.prod(shape);storage=torch.full((17+count+23,),fill,dtype=torch.float32)
    return storage[17:17+count].reshape(shape),storage
def frame(storage,before,view=None):
    mask=torch.zeros_like(storage,dtype=torch.bool)
    if view is not None:mask[view.storage_offset():view.storage_offset()+view.numel()]=True
    mask=mask.repeat_interleave(4)
    assert torch.equal(storage.view(torch.uint8)[~mask],before[~mask])
def dense(q,k,v,seed):
    B,H,T,K=q.shape;V=v.shape[-1]
    rate=(1-torch.exp2(-5-torch.arange(H,dtype=q.dtype))).view(1,H,1,1)
    state=seed;outs=[];hist=[]
    for t in range(T):
        if t%64==0:hist.append(state)
        state=rate*state+k[:,:,t,:,None]*v[:,:,t,None,:]
        outs.append(torch.einsum('bhk,bhkv->bhv',q[:,:,t],state)*(K**-.5))
    out=torch.stack(outs,2) if outs else torch.empty((B,H,0,V))
    return out,state,hist

def run(B,H,T,K,V,seed=True,final=True,incoming=True,terminal=True):
    BT,BK,BV,NT,NK,NV,_=m._plan(B,H,T,K,V);scale=K**-.5
    q,qb=tensor((B,H,T,K));k,kb=tensor(q.shape);v,vb=tensor((B,H,T,V))
    si,sb=tensor((B,H,K,V));gi,gb=tensor(v.shape);gt,tb=tensor(si.shape)
    for x in (q,k,v,si,gi,gt):x.copy_(torch.randn_like(x)*.1)
    if not seed:si.zero_()
    if not incoming:gi.zero_()
    if not terminal or not final:gt.zero_()
    h,hb=tensor((B,H,NT*K,V));o,ob=tensor(v.shape);f,fb=tensor(si.shape)
    dh,dhb=tensor(h.shape);dq,dqb=tensor(q.shape);dk,dkb=tensor(k.shape);parts,pb=tensor((NK,*v.shape))
    initial=[x.view(torch.uint8).clone() for x in (qb,kb,vb,sb,gb,tb,hb,ob,fb,dhb,dqb,dkb,pb)]
    strides=(T*K,K,1,T*V,V,1,NT*K*V,V)
    if B*H:
        m.chunk_retention_fwd_kernel_h[(NK,NV,B*H)](k,v,h,si if seed else None,f if final else None,*strides,
            H=H,T=T,K=K,V=V,BT=BT,BK=BK,BV=BV,NT=NT,USE_INITIAL_STATE=seed,STORE_FINAL_STATE=final,num_warps=4 if BK==64 else 2,num_stages=1)
        if NT:m.chunk_retention_fwd_kernel_o[(NV,NT,B*H)](q,k,v,h,o,*strides,scale,H=H,T=T,K=K,V=V,BT=BT,BK=BK,BV=BV,num_warps=4 if BK==64 else 2,num_stages=1)
    forward_frames=[x.view(torch.uint8).clone() for x in (hb,ob,fb)]
    if B*H and NT:
        m.chunk_retention_bwd_kernel_dh[(NK,NV,B*H)](q,gi,dh,gt if terminal and final else None,*strides,scale,
            H=H,T=T,K=K,V=V,BT=BT,BK=BK,BV=BV,NT=NT,USE_FINAL_GRAD=terminal and final,num_warps=4 if BK==64 else 2,num_stages=1)
        m.chunk_retention_bwd_kernel_dqkv[(NK,NT,B*H)](q,k,v,h,gi,dh,dq,dk,parts,*strides,scale,
            H=H,T=T,K=K,V=V,BT=BT,BK=BK,BV=BV,NT=NT,num_warps=4 if BK==64 else 2,num_stages=1)
    dv=parts.sum(0)
    rq=q.clone().requires_grad_(True);rk=k.clone().requires_grad_(True);rv=v.clone().requires_grad_(True)
    ro,rf,rh=dense(rq,rk,rv,si)
    loss=(ro*gi).sum()+(rf*gt).sum()+0*(rq.sum()+rk.sum()+rv.sum())
    gradients=torch.autograd.grad(loss,(rq,rk,rv))
    torch.testing.assert_close(o,ro,rtol=.004,atol=.0001)
    if final:torch.testing.assert_close(f,rf,rtol=.004,atol=.0001)
    for i,state in enumerate(rh):torch.testing.assert_close(h[:,:,i*K:(i+1)*K],state,rtol=.004,atol=.0001)
    for actual,expected in zip((dq,dk,dv),gradients):torch.testing.assert_close(actual,expected,rtol=.006,atol=.00015)
    # Independent future-state adjoints at each chunk endpoint.
    rate=(1-torch.exp2(-5-torch.arange(H,dtype=q.dtype))).view(1,H,1,1)
    future=gt.clone();expected_dh={T:future.clone()}
    for t in range(T-1,-1,-1):
        future=rate*(future+scale*q[:,:,t,:,None]*gi[:,:,t,None,:])
        expected_dh[t]=future.clone()
    for i in range(NT):torch.testing.assert_close(dh[:,:,i*K:(i+1)*K],expected_dh[min((i+1)*64,T)],rtol=.005,atol=.0001)
    # Each partial is the V gradient from that independent feature subset.
    for block in range(NK):
        lo=block*BK;hi=min(lo+BK,K)
        pv=rv.detach().clone().requires_grad_(True)
        qo,qs,_=dense(q[:,:,:,lo:hi],k[:,:,:,lo:hi],pv,si[:,:,lo:hi])
        # dense uses subset width for scale; restore original K scale.
        qo=qo*(scale/((hi-lo)**-.5))
        ploss=(qo*gi).sum()+(qs*gt[:,:,lo:hi]).sum()+pv.sum()*0
        expected=torch.autograd.grad(ploss,pv)[0]
        torch.testing.assert_close(parts[block],expected,rtol=.006,atol=.00015)
    stores=(qb,kb,vb,sb,gb,tb,hb,ob,fb,dhb,dqb,dkb,pb)
    for x,before in zip(stores[:6],initial[:6]):frame(x,before)
    for x,before in zip((hb,ob,fb),forward_frames):frame(x,before)
    for x,before,view in zip(stores[6:],initial[6:],(h,o,f if final else None,dh,dq,dk,parts)):frame(x,before,view)
    records.append((B,H,T,K,V,seed,final,incoming,terminal))
for t,k,v in ((0,1,1),(1,1,7),(63,15,17),(64,16,32),(65,17,33),(129,65,67)):
    run(1,2,t,k,v)
for seed,final,incoming,terminal in ((False,False,True,False),(True,False,False,False),(False,True,False,True),(True,True,True,False)):
    run(1,1,65,33,17,seed,final,incoming,terminal)
run(2,3,1,17,7)
run(0,2,65,17,7)
print(json.dumps(dict(cases=len(records),all_outputs_and_scratch=True,independent_recurrence_and_autograd=True,full_byte_frames=True,gpu_execution=False)))
