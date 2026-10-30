"""Actual private Python host and Triton interpreter controls; no GPU claims."""
from pathlib import Path
import os
D=Path(__file__).resolve().parent
os.environ.update(TRITON_INTERPRET='1',TRITON_CACHE_DIR='/tmp/parallel-attention-cpu-cache',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import hashlib,importlib.util,json,time,traceback,sys
sys.dont_write_bytecode=True
import torch
P=D/'parallel_attention_repaired.py'
spec=importlib.util.spec_from_file_location('private_parallel_attention_cpu',P)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(1)
from contextlib import nullcontext
# Only CUDA residency and device context are shimmed. All Tensor layouts,
# storage queries, kernels, allocations, reductions and autograd execute.
old_cuda=torch.Tensor.is_cuda;old_device=torch.cuda.device
torch.Tensor.is_cuda=property(lambda _:True)
contexts=[]
def device_context(device):
    contexts.append(str(device));return nullcontext()
torch.cuda.device=device_context
launches=[]
class Spy:
    def __init__(self,name,inner):self.name,self.inner=name,inner
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            launches.append({'kernel':self.name,'grid':grid,'strides':list(args[5:11] if self.name=='forward' else args[8:14]),'scale':args[11] if self.name=='forward' else args[14],'BK':kwargs['BK'],'BV':kwargs['BV']})
            return self.inner[grid](*args,**kwargs)
        return call
m.parallel_rebased_fwd_kernel=Spy('forward',m.parallel_rebased_fwd_kernel)
m.parallel_rebased_bwd_kernel=Spy('backward',m.parallel_rebased_bwd_kernel)
def reference(q,k,v,scaled,normalized,both,eps):
    T,K=q.shape[-2:]
    scores=torch.einsum('bhte,bhje->bhtj',q,k)*(K**-.5 if scaled else 1)
    weights=torch.where(torch.arange(T)[:,None]>=torch.arange(T)[None,:],scores.square(),0.)
    o=torch.einsum('bhtj,bhjp->bhtp',weights,v);z=weights.sum(-1)
    return (o,z) if both else (o/(z[...,None]+eps) if normalized else o)
cases=[('tiny',(1,1,5,1,1),True,False,True),('ragged',(1,2,33,17,19),False,False,True),('interchunk',(1,1,137,17,23),True,False,True),('value_shards',(1,1,65,127,131),True,False,True),('singleton_stride',(2,1,17,17,19),False,False,True),('normalized',(1,2,33,17,19),True,True,False),('raw',(1,1,65,17,19),False,False,False),('both_priority',(1,1,7,3,5),True,True,True),('expanded_cotangent',(1,1,33,17,19),True,False,True),('storage_offset',(1,1,5,3,5),True,False,True),('alias',(1,1,5,3,3),False,False,True)]
records=[]
try:
    for idx,(name,shape,scaled,normalized,both) in enumerate(cases):
        start=time.monotonic();B,H,T,K,V=shape
        gen=torch.Generator().manual_seed(840+idx)
        def tensor(d):
            if name=='singleton_stride':return (torch.randn(B,T,H,d,generator=gen)*.2).permute(0,2,1,3).requires_grad_()
            return (torch.randn(B,H,T,d,generator=gen)*.2).requires_grad_()
        q,k,v=tensor(K),tensor(K),tensor(V)
        if name=='storage_offset':
            def offset(x):
                base=torch.empty(x.numel()+7);base[7:]=x.detach().flatten()
                return base[7:].view(x.shape).requires_grad_()
            q,k,v=[offset(x) for x in (q,k,v)]
        if name=='alias':k=q.detach().requires_grad_()
        saved_bytes=[x.untyped_storage().clone() for x in (q,k,v)]
        qr,kr,vr=[x.detach().double().requires_grad_() for x in (q,k,v)]
        go=(torch.randn(B,H,V,T,generator=gen)*.1).transpose(-1,-2)
        gz=(torch.randn(B,T,H,generator=gen)*.1).transpose(-1,-2)
        if name=='expanded_cotangent':go=torch.ones(1).expand(B,H,T,V);gz=torch.ones(1).expand(B,H,T)
        begin=len(launches);eps=.01
        actual=m.parallel_rebased(q,k,v,eps,scaled,normalized,both)
        expected=reference(qr,kr,vr,scaled,normalized,both,eps)
        ys=actual if both else (actual,);rs=expected if both else (expected,)
        cot=(go,gz) if both else (go,)
        grads=torch.autograd.grad(ys,(q,k,v),cot)
        rgrads=torch.autograd.grad(rs,(qr,kr,vr),tuple(x.double() for x in cot))
        diffs=[]
        for value,ref in zip((*ys,*grads),(*rs,*rgrads)):
            assert value.shape==ref.shape and value.dtype==torch.float32
            assert all(value is not x and value.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for x in (q,k,v))
            err=(value.double()-ref).abs();diffs.append(0. if not err.numel() else float(err.max()))
            torch.testing.assert_close(value.double(),ref,atol=3e-5,rtol=7e-4)
        assert len(launches)-begin==2
        assert all(bytes(x.untyped_storage())==bytes(b) for x,b in zip((q,k,v),saved_bytes))
        for launch in launches[begin:]:assert launch['strides']==[T*K,K,1,T*V,V,1]
        records.append({'name':name,'shape':shape,'use_scale':scaled,'use_normalize':normalized,'return_both':both,'input_strides':[list(x.stride()) for x in (q,k,v)],'incoming_cotangent_strides':[list(x.stride()) for x in cot],'max_abs_errors':diffs,'launches':launches[begin:],'seconds':time.monotonic()-start,'passed':True})
        print(json.dumps({'case':name,'passed':True,'errors':diffs}),flush=True)
    empty_records=[]
    for shape in [(0,2,3,4,5),(2,0,3,4,5),(2,2,0,4,5)]:
        B,H,T,K,V=shape
        for both,normalize in [(True,True),(False,True),(False,False)]:
            q,k,v=[torch.empty(B,H,T,d,requires_grad=True) for d in (K,K,V)]
            start=len(launches)
            y=m.parallel_rebased(q,k,v,return_both=both,use_normalize=normalize)
            ys=y if both else (y,)
            grads=torch.autograd.grad(ys,(q,k,v),tuple(torch.ones_like(x) for x in ys))
            assert len(launches)==start and all(x.numel()==0 for x in (*ys,*grads))
            empty_records.append({'shape':shape,'both':both,'normalize':normalize,'no_launch':True})
    q,k,v=[torch.empty_strided((0,2,3,d),(100,50,10,2),requires_grad=True) for d in (4,4,5)]
    start=len(launches)
    ys=m.parallel_rebased(q,k,v,return_both=True)
    grads=torch.autograd.grad(ys,(q,k,v),tuple(torch.ones_like(x) for x in ys))
    assert len(launches)==start and all(g.stride()==x.stride() for g,x in zip(grads,(q,k,v)))
    empty_records.append({'shape':[0,2,3,4,5],'noncanonical_empty_strides':[100,50,10,2],'empty_like_preserves_strides':True,'no_launch':True})
    backward_autograd_rejections=[]
    for empty in [False,True]:
        for which in ['do','dz']:
            q,k,v=[torch.ones(0 if empty else 1,1,1,1,requires_grad=True) for _ in range(3)]
            ys=m.parallel_rebased(q,k,v,return_both=True)
            cot=[torch.ones_like(y) for y in ys]
            cot[0 if which=='do' else 1]=torch._neg_view(cot[0 if which=='do' else 1])
            start=len(launches)
            try:torch.autograd.grad(ys,(q,k,v),tuple(cot))
            except ValueError as error:assert 'resolved cotangent' in str(error)
            else:raise AssertionError('autograd lazy-negative guard failed')
            assert len(launches)==start
            backward_autograd_rejections.append({'empty':empty,'cotangent':which,'rejected_before_backward_launch':True})
    q=torch.ones(1,1,3,4);v=torch.ones(1,1,3,5)
    bad=[('dtype',(q.half(),q.half(),v.half())),('shape',(q,q[:,:,:2],v)),('layout',(q.transpose(-1,-2),q.transpose(-1,-2),torch.ones(1,1,4,5))),('K_limit',(torch.ones(1,1,3,129),torch.ones(1,1,3,129),v)),('lazy_neg',(torch._neg_view(q),q,v)),('non_tensor',(None,q,v))]
    rejected=[]
    for label,args in bad:
        try:m.parallel_rebased(*args)
        except (ValueError,TypeError):rejected.append(label)
        else:raise AssertionError('guard failed: '+label)
    for kwargs in [{'use_scale':1},{'use_normalize':0},{'return_both':1},{'eps':True},{'eps':torch.tensor(1.)},{'eps':10**1000}]:
        try:m.parallel_rebased(q,q,v,**kwargs)
        except (TypeError,ValueError):rejected.append(str(list(kwargs)))
        else:raise AssertionError('scalar guard failed')
except Exception:
    records.append({'passed':False,'failure':traceback.format_exc()});raise
finally:
    receipt={'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mode':'Actual wrapper and Triton CPU interpreter with CUDA residency/context observation shims only','GPU_execution':False,'universal_numerical_proof':False,'torch_version':torch.__version__,'triton_version':m.triton.__version__,'cases':records,'empty_cases':locals().get('empty_records',[]),'owning_device_context_entries':len(contexts),'guard_rejections':locals().get('rejected',[]),'backward_autograd_rejections':locals().get('backward_autograd_rejections',[]),'all_passed':len(records)==len(cases) and all(x['passed'] for x in records) and len(locals().get('rejected',[]))==12 and len(locals().get('backward_autograd_rejections',[]))==4}
    (D/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')

    torch.Tensor.is_cuda=old_cuda
    torch.cuda.device=old_device
