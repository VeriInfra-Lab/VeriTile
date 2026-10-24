"""CPU interpreter/guard checks. This does not validate GPU floating instructions."""
import os
os.environ['TRITON_INTERPRET']='1'
import importlib.util, contextlib, json, torch
from pathlib import Path
path=Path(__file__).with_name('token_attn_llama2_repaired.py')
spec=importlib.util.spec_from_file_location('checked_llama2',path)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
class CudaFacade(torch.Tensor):
    @property
    def is_cuda(self): return True

def facade(x): return x.as_subclass(CudaFacade)
torch.cuda.device=lambda _:contextlib.nullcontext()
torch.manual_seed(571)
count=0

def case(D=16,M=33,B=3,H=6,KV=2):
    N=5; T=max(7,B*(M+2))
    q=facade(torch.randn(B,H,D)); k=facade(torch.randn(N,KV,D))
    out=facade(torch.randn(H,T)); loc=facade(torch.full((B,M),-97,dtype=torch.int32))
    starts=facade(torch.tensor([b*(M+2) for b in range(B)],dtype=torch.int32))
    lengths=facade(torch.tensor([max(0,M-b) for b in range(B)],dtype=torch.int32))
    for b,length in enumerate(lengths.tolist()):
        if length: loc[b,M-length:]=torch.arange(length,dtype=torch.int32)%N
    return [q,k,out,loc,starts,lengths,M]

def run(args):
    global count
    before=[x.clone() for x in args[:-1]]
    result=mod.token_att_fwd(*args)
    assert result is None
    q,k,out,loc,starts,lens,M=args
    ref=before[2].clone()
    for b,L in enumerate(lens.tolist()):
        for h in range(q.shape[1]):
            if L:
                expected=(q[b,h].double()[None,:]*k[loc[b,M-L:].long(),h//(q.shape[1]//k.shape[1])].double()).sum(1)/(q.shape[-1]**.5)
                ref[h,starts[b]:starts[b]+L]=expected.float()
    torch.testing.assert_close(out,ref,atol=2e-6,rtol=2e-5)
    for i in [0,1,3,4,5]: assert torch.equal(args[i],before[i])
    count+=1

for D in [16,32,64,128]:
    for M in [0,1,31,32,33,65]: run(case(D,M))
    run(case(D,33,B=0))
    a=case(D,33); a[5].zero_(); a[1]=facade(torch.empty(0,2,D)); run(a)
# Inactive prefix entries may be negative; only active suffix indices are loaded.
a=case(); a[5][:]=1; run(a)
# Read buffers may overlap; output remains separate.
a=case(B=3,H=2,KV=2); a[1]=a[0]; a[3][:]=0; run(a)

def reject(mut,etype=ValueError):
    global count
    a=case(); mut(a)
    before=a[2].clone()
    try: mod.token_att_fwd(*a)
    except etype: pass
    else: raise AssertionError('accepted invalid input')
    assert torch.equal(a[2],before)
    count+=1
reject(lambda a:a.__setitem__(0,a[0].as_subclass(torch.Tensor)))
reject(lambda a:a.__setitem__(0,a[0].double()),TypeError)
reject(lambda a:a.__setitem__(3,a[3].long()),TypeError)
reject(lambda a:a.__setitem__(0,torch._neg_view(a[0])))
reject(lambda a:a.__setitem__(0,a[0].transpose(0,1)))
reject(lambda a:a.__setitem__(6,True))
reject(lambda a:a[5].__setitem__(0,-1))
reject(lambda a:a[5].__setitem__(0,34))
reject(lambda a:a[4].__setitem__(0,-1))
reject(lambda a:a[4].__setitem__(0,a[2].shape[1]))
reject(lambda a:a[4].__setitem__(1,0))
reject(lambda a:a[3].__setitem__((0,32),-1))
reject(lambda a:a[3].__setitem__((0,32),5))
# FP32 output physically overlapping Q, with a legal dense output shape.
def alias(a): a[2]=a[0].reshape(6,-1)
reject(alias)
# Ordinary byte-offset storage can be four-byte unaligned even in a dense view.
def unaligned(a):
    shape=a[0].shape
    storage=torch.UntypedStorage(a[0].numel()*4+1)
    a[0]=facade(torch.empty(0,dtype=torch.float32).set_(storage[1:],0,shape))
reject(unaligned)
# Mixed FP32/int32 input storage is rejected even with valid zero page IDs.
def mixed_alias(a):
    a[0].zero_()
    a[3]=a[0].reshape(-1)[:a[3].numel()].view(torch.int32).reshape(a[3].shape)
reject(mixed_alias)
# Torch's [1,0] table has stride [1,1]; the empty branch must accept it.
run(case(M=0,B=1,H=1,KV=1))
print(json.dumps({'cases':count,'status':'pass','backend':'CPU Triton interpreter and CUDA metadata facade; no GPU validation'}))
