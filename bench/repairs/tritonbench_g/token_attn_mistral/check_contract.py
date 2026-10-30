"""CPU interpreter/guard checks. This does not validate GPU floating instructions."""
import os
os.environ['TRITON_INTERPRET']='1'
import torch,contextlib,importlib.util,json
from pathlib import Path
class CudaFacade(torch.Tensor):
    @property
    def is_cuda(self): return True
def f(x):return x.as_subclass(CudaFacade)
torch.cuda.device=lambda _:contextlib.nullcontext()
torch.manual_seed(571)
report=[]
for name in ['token_attn_mistral']:
 sliding=name.endswith('mistral')
 p=Path(__file__).with_name(name+'_repaired.py')
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 count=0
 def case(D=16,L=129,B=3,window=65,att=None):
  H=6;KV=2;R=max(B,1);N=7;M=max(L,1);T=max(1,B*(L+5))
  P=f(torch.randn(H,T));V=f(torch.randn(N,KV,D));out=f(torch.randn(B,H,D))
  table=f(torch.arange(R*M,dtype=torch.int32).reshape(R,M)%N)
  req=f(torch.arange(B,dtype=torch.int32));seq=f(torch.tensor([max(L-b,0) for b in range(B)],dtype=torch.int32))
  starts=f(torch.arange(B,dtype=torch.int32)*(L+5))
  if sliding:
   unused=f(torch.full((B,),-2147483648,dtype=torch.int32))
   lengths=f(torch.tensor([max(0,(L if att is None else att)-b) for b in range(B)],dtype=torch.int32))
   return [P,V,out,table,req,unused,seq,starts,lengths,window]
  return [P,V,out,table,req,starts,seq]
 def run(a):
  nonlocal_dummy=None
  P,V,out,table,req,_,seq=a[:7]
  starts,lengths=(a[7],a[8]) if sliding else (a[5],a[6])
  before=[x.clone() for x in a if isinstance(x,torch.Tensor)]
  assert mod.token_att_fwd2(*a) is None
  ref=torch.zeros_like(out)
  for b in range(out.shape[0]):
   begin=max(int(seq[b])-a[9],0) if sliding else 0
   limit=min(int(lengths[b]),int(seq[b])-begin)
   for h in range(P.shape[0]):
    if limit:
     vals=V[table[req[b],begin:begin+limit].long(),h//(P.shape[0]//V.shape[1])].double()
     ref[b,h]=(P[h,starts[b]:starts[b]+limit].double()[:,None]*vals).sum(0).float()
  torch.testing.assert_close(out,ref,atol=5e-5,rtol=3e-5)
  for i,x in enumerate([x for x in a if isinstance(x,torch.Tensor)]):
   if i!=2: assert torch.equal(x,before[i])
 for D in [16,32,64,128]:
  for L in [0,1,127,128,129,257]:run(case(D,L));count+=1
  run(case(D,129,B=0));count+=1
  a=case(D,0);a[1]=f(torch.empty(0,2,D));run(a);count+=1
 if sliding:
  for L,window,att in [(129,0,5),(129,1,129),(129,65,1),(129,256,1),(129,256,257)]:
   a=case(L=L,window=window,att=att); a[0]=f(torch.randn(6,3*(att+5)));a[7]=f(torch.arange(3,dtype=torch.int32)*(att+5));run(a);count+=1
  a=case(L=129,window=256,att=1,B=1)
  a[3].zero_();a[3][0,1]=1;a[1][1]=float('nan')
  mod.token_att_fwd2(*a);assert torch.isnan(a[2]).all();count+=1
 else:
  a=case(L=1,B=1);a[3].zero_();a[1][1]=float('nan');mod.token_att_fwd2(*a);assert torch.isfinite(a[2]).all();count+=1
 def reject(mut,typ=ValueError):
  a=case();mut(a);before=a[2].clone()
  try:mod.token_att_fwd2(*a)
  except typ:pass
  else:raise AssertionError('accepted invalid input')
  assert torch.equal(a[2],before)
 reject(lambda a:a.__setitem__(0,a[0].as_subclass(torch.Tensor)));count+=1
 reject(lambda a:a.__setitem__(0,a[0].double()),TypeError);count+=1
 reject(lambda a:a.__setitem__(3,a[3].long()),TypeError);count+=1
 reject(lambda a:a.__setitem__(0,torch._neg_view(a[0])));count+=1
 reject(lambda a:a[4].__setitem__(0,-1));count+=1
 reject(lambda a:a[4].__setitem__(0,3));count+=1
 reject(lambda a:a[6].__setitem__(0,-1));count+=1
 reject(lambda a:a[6].__setitem__(0,130));count+=1
 reject(lambda a:a[7 if sliding else 5].__setitem__(0,-1));count+=1
 reject(lambda a:a[8 if sliding else 6].__setitem__(0,9999));count+=1
 reject(lambda a:a[3].__setitem__((0,128),-1));count+=1
 reject(lambda a:a[3].__setitem__((0,128),7));count+=1
 def alias(a): a[2]=a[0].flatten()[:3*6*16].view(3,6,16)
 reject(alias);count+=1
 def unaligned(a):
  shape=a[0].shape;storage=torch.UntypedStorage(a[0].numel()*4+1)
  a[0]=f(torch.empty(0,dtype=torch.float32).set_(storage[1:],0,shape))
 reject(unaligned);count+=1
 if sliding:
  reject(lambda a:a.__setitem__(9,-1));count+=1
  reject(lambda a:a.__setitem__(9,True));count+=1
 def mixed_alias(a):
  a[0].zero_()
  a[3]=a[0].reshape(-1)[:a[3].numel()].view(torch.int32).reshape(a[3].shape)
 reject(mixed_alias);count+=1
 report.append({'task':name,'cases':count,'status':'pass'})
print(json.dumps({'checks':report,'backend':'CPU Triton interpreter with CUDA metadata facade; no GPU numerical validation'}))
