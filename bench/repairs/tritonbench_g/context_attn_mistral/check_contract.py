"""Checked wrapper and source-graph mapping; CPU interpretation is not GPU validation."""
import os
os.environ['TRITON_INTERPRET']='1'
import torch, contextlib, importlib.util, json, ast
from pathlib import Path
path=Path(__file__).with_name('context_attn_mistral_repaired.py')
spec=importlib.util.spec_from_file_location('checked_context_mistral',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class CudaFacade(torch.Tensor):
 @property
 def is_cuda(self):return True
def f(t):return t.as_subclass(CudaFacade)
torch.cuda.device=lambda _:contextlib.nullcontext()
torch.manual_seed(571)
count=0

def case(D=16,L=3,B=2,maximum=None,window=2,H=2,KV=1):
 T=B*(L+2)
 q=f(torch.randn(T,H,D)*0.12);k=f(torch.randn(T,KV,D)*0.12);v=f(torch.randn(T,KV,D))
 out=f(torch.randn(T,H,D))
 start=f(torch.arange(B,dtype=torch.int32)*(L+2))
 seq=f(torch.tensor([max(0,L-b) for b in range(B)],dtype=torch.int32))
 return [q,k,v,out,start,seq,L if maximum is None else maximum,window]

def run(a):
 global count
 q,k,v,out,start,seq,maximum,window=a
 before=[x.clone() for x in a[:6]]
 assert mod.context_attention_fwd_repaired(*a) is None
 ref=before[3].clone()
 for b,length in enumerate(seq.tolist()):
  for h in range(q.shape[1]):
   for row in range(min(length,((maximum+127)//128)*128)):
    stop=(row//128+1)*128
    j=torch.arange(stop)
    ids=start[b]+j[:min(length,stop)]
    keys=torch.zeros(stop,q.shape[2]);vals=torch.zeros_like(keys)
    keys[:len(ids)]=k[ids,h//(q.shape[1]//k.shape[1])]
    vals[:len(ids)]=v[ids,h//(q.shape[1]//k.shape[1])]
    score=(keys.double()*q[start[b]+row,h].double()).sum(1)*torch.tensor(q.shape[2]**(-.5),dtype=torch.float32).item()
    # For bounded input scores, exp(-1e9) underflows in native FP32.
    allowed=(j<=row)&((row-j)<min(max(window,0),2**63-1))
    if allowed.any():
     p=torch.softmax(torch.where(allowed,score,torch.full_like(score,-1e9)),0)
     ref[start[b]+row,h]=(p[:,None]*vals.double()).sum(0).float()
    else:ref[start[b]+row,h]=0
 torch.testing.assert_close(out,ref,atol=5e-6,rtol=3e-5)
 for i in [0,1,2,4,5]:assert torch.equal(a[i],before[i])
 count+=1
for d in [16,32,64,128]:
 for length in [0,1,3,129]:run(case(d,length,B=1 if length==129 else 2))
 run(case(d,3,B=0));run(case(d,3,maximum=0))
for w in [-10,0,1,128,2**80]:run(case(window=w))
run(case(L=129,B=1,maximum=1,window=65))
# Read-only floating views may alias one another.
a=case(H=1,KV=1);a[1]=a[0];a[2]=a[0];run(a)

# Live launch with zero-length metadata still has no data stores.
run(case(L=0,maximum=128))
a=case(L=0,B=1,maximum=1)
a[:4]=[f(torch.empty(0,2,16)),f(torch.empty(0,1,16)),f(torch.empty(0,1,16)),f(torch.empty(0,2,16))]
run(a)
# The raw-word reader supports mixed-dtype read aliases without a tagged-cell conflict.
a=case(B=1,H=1,KV=1);a[0].reshape(-1)[0]=0
a[4]=a[0].reshape(-1)[:1].view(torch.int32)
run(a)

def reject(mut,typ=ValueError):
 global count
 a=case();mut(a);before=a[3].clone()
 try:mod.context_attention_fwd_repaired(*a)
 except typ:pass
 else:raise AssertionError('invalid input accepted')
 assert torch.equal(a[3],before);count+=1
reject(lambda a:a.__setitem__(0,a[0].as_subclass(torch.Tensor)))
reject(lambda a:a.__setitem__(0,a[0].double()),TypeError)
reject(lambda a:a.__setitem__(4,a[4].long()),TypeError)
reject(lambda a:a.__setitem__(0,torch._neg_view(a[0])))
reject(lambda a:a.__setitem__(6,True))
reject(lambda a:a.__setitem__(6,-1))
reject(lambda a:a.__setitem__(7,1.5))
reject(lambda a:a[4].__setitem__(0,-1))
reject(lambda a:a[5].__setitem__(0,-1))
reject(lambda a:a[5].__setitem__(0,11))
reject(lambda a:a[4].__setitem__(1,0))
reject(lambda a:a.__setitem__(3,a[0]))
def unaligned(a):
 shape=a[0].shape;storage=torch.UntypedStorage(a[0].numel()*4+1)
 a[0]=f(torch.empty(0,dtype=torch.float32).set_(storage[1:],0,shape))
reject(unaligned)
# Structural source-to-recurrence audit: these are the only floating equality guards.
tree=ast.parse(path.read_text());kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fwd_kernel_repaired')
floating_eq=[n for n in ast.walk(kernel) if isinstance(n,ast.Compare) and any(isinstance(op,ast.Eq) for op in n.ops)]
assert {ast.unparse(n) for n in floating_eq}=={'m_ij == -1000000000.0','l_i_new == 0.0'}
body=ast.unparse(kernel)
for node in ['qk += tl.dot(q, k)','qk *= sm_scale','p = tl.exp(qk - m_ij[:, None])','l_i_new = alpha * l_i + beta * l_ij','acc_scale = l_i / l_i_new * alpha','p = p.to(v.dtype)','acc += tl.dot(p, v)','m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float(\'inf\')']:
 assert node in body,node
count+=1
print(json.dumps({'cases':count,'status':'pass','backend':'CPU interpreter and CUDA metadata facade; no GPU numerical validation'}))
