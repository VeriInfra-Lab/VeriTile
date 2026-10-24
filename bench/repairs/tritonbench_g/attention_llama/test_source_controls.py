"""Actual Triton source CPU interpreter controls, not GPU conformance."""
import ast
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'attention_llama_repaired.py'
ORIGINAL = HERE.parents[3] / 'bench/tritonbench_g/attention_llama/attention_llama.py'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mod = load(SOURCE, 'llama_repaired')
rows = []
torch.manual_seed(901)

def strided(shape, dtype, fill=0):
    b,h,n,d=shape
    strides=(h*(n+3)*(d*2+5),(n+3)*(d*2+5),d*2+5,2)
    size=17+sum((s-1)*t for s,t in zip(shape,strides))+1+19
    storage=torch.full((size,),fill,dtype=dtype)
    return storage.as_strided(shape,strides,17),storage

def run(label, H=3, N=65, D=16, fp8=False, causal=False, scale=.25, alias=False, uniform=False, kernel=None):
    B,NH=1,2
    q,qb=strided((B,NH,H,D),torch.float16)
    k,kb=strided((B,NH,N,D),torch.int8 if fp8 else torch.float16)
    v,vb=(k,kb) if alias else strided((B,NH,N,D),k.dtype)
    o,ob=strided((B,NH,H,D),torch.float16,919.)
    q.copy_(torch.zeros_like(q) if uniform else torch.randn_like(q)*.1)
    kk=torch.ones(k.shape,dtype=torch.float16) if uniform else torch.randn(k.shape,dtype=torch.float16)*.2
    vv=torch.ones(v.shape,dtype=torch.float16) if uniform else torch.randn(v.shape,dtype=torch.float16)*.2
    if fp8:
        k.copy_(kk.to(torch.float8_e5m2).view(torch.int8))
        if not alias: v.copy_(vv.to(torch.float8_e5m2).view(torch.int8))
        # Raw negative zero is finite and must stay byte-identical.
        if not uniform: k[0,0,0,0]=-128
    else:
        k.copy_(kk)
        if not alias:v.copy_(vv)
    saved=[t.clone() for t in (qb,kb,vb,ob)]
    (kernel or mod._fwd_kernel)[((H+63)//64,B*NH)](
        q,k,v,scale,o,*q.stride(),*k.stride(),*v.stride(),*o.stride(),NH,H,N,
        start_position=0,IS_CAUSAL=causal,BLOCK_M=64,BLOCK_N=64,BLOCK_DMODEL=D,
        USE_FP8=fp8,num_warps=4 if D<=64 else 8,num_stages=2)
    kreal=k.view(torch.float8_e5m2).float() if fp8 else k.float()
    vreal=v.view(torch.float8_e5m2).float() if fp8 else v.float()
    scores=(q.float()@kreal.transpose(-1,-2))*scale
    if causal: scores.masked_fill_(torch.arange(N)[None,:]>torch.arange(H)[:,None],-torch.inf)
    expected=scores.softmax(-1)@vreal
    if kernel is None:torch.testing.assert_close(o.float(),expected,rtol=.006,atol=.003)
    for old,new in zip(saved,(qb,kb,vb,ob)):
        if new is ob:continue
        assert torch.equal(old,new)
    allowed=torch.zeros_like(ob,dtype=torch.bool)
    allowed.as_strided(o.shape,o.stride(),o.storage_offset()).fill_(True)
    assert torch.equal(ob[~allowed],saved[-1][~allowed])
    rows.append(dict(case=label,status='PASS',H=H,N=N,D=D,fp8=fp8,causal=causal,frame=True))
    return o

for fp8 in (False,True):
    for causal in (False,True):
        for H,N in ((1,1),(3,63),(63,64),(65,65),(2,127)):
            run('tail-layout',H,N,fp8=fp8,causal=causal,alias=H==65)
for dim in (16,32,64,128):run('head-profile',D=dim)
for scale in (0.,-0.5):
    for fp8 in (False,True):run('signed-scale',fp8=fp8,scale=scale,uniform=True)

# Original defect witness remains separate: partial second block keeps padded scores.
prefix=ORIGINAL.read_text().split('################################################################')[0]
with tempfile.TemporaryDirectory() as tmp:
    oldpath=Path(tmp)/'llama_original.py';oldpath.write_text(prefix)
    old=load(oldpath,'llama_original')
    result=run('original-N65-defect',H=1,N=65,uniform=True,kernel=old._fwd_kernel)
    torch.testing.assert_close(result,torch.full_like(result,65/128),rtol=0,atol=0)
fixed=run('repaired-N65-uniform',H=1,N=65,uniform=True)
torch.testing.assert_close(fixed,torch.ones_like(fixed),rtol=0,atol=0)
print(json.dumps(dict(cases=len(rows),all_output_cells=True,frames=True,gpu_execution=False)))
