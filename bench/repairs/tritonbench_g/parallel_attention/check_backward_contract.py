"""Focused actual-Torch/source-boundary controls; no GPU execution claim."""
from pathlib import Path
import ast,ctypes,hashlib,importlib.util,json,sys
from contextlib import nullcontext
from types import SimpleNamespace
import torch
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
P=D/'parallel_attention_repaired.py'
O=D.parents[2]/'tritonbench_g/parallel_attention/parallel_attention.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('parallel_backward_controls',P)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
original=ast.parse(O.read_text())
cls=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='ParallelBasedFunction')
backward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
backward.decorator_list=[]
module=ast.Module(body=[backward],type_ignores=[]);ast.fix_missing_locations(module)
class AtLaunch(Exception):pass
captured=[]
class Capture:
    def __getitem__(self,grid):
        def launch(*args,**kwargs):captured.append((args,grid));raise AtLaunch
        return launch
ns={'torch':torch,'triton':m.triton,'parallel_rebased_bwd_kernel':Capture()}
exec(compile(module,str(O),'exec'),ns)
old_cuda=torch.Tensor.is_cuda;old_device=torch.cuda.device
old_contiguous=torch.Tensor.contiguous;old_launch=m.parallel_rebased_bwd_kernel
old_conj=torch.Tensor.is_conj;old_tensor_device=torch.Tensor.device
copies=[]
def track_copy(t,*a,**kw):copies.append(id(t));return old_contiguous(t,*a,**kw)
torch.Tensor.is_cuda=property(lambda _:True)
torch.cuda.device=lambda _:nullcontext()
m.parallel_rebased_bwd_kernel=Capture()
q=torch.ones(1,1,1,1);v=q.clone();do=torch.ones_like(v);dz=torch.ones(1,1,1)
ctx=SimpleNamespace(saved_tensors=(q,q,v),scale=1.)
rejected=[];witnesses=[];observed=[]
def reject(name,fn):
    calls=len(captured);count=len(copies)
    try:fn()
    except (TypeError,ValueError) as e:rejected.append({'name':name,'error':str(e)})
    else:raise AssertionError('guard accepted '+name)
    assert len(captured)==calls and len(copies)==count
try:
    for which in ['do','dz']:
        go=torch._neg_view(do) if which=='do' else do
        gz=torch._neg_view(dz) if which=='dz' else dz
        lazy=go if which=='do' else gz
        ready=lazy.contiguous()
        assert ready is lazy and ready.is_neg() and ready.data_ptr()==lazy.data_ptr()
        raw=ctypes.c_float.from_address(ready.data_ptr()).value
        logical=ready.item()
        assert (raw,logical)==(1.,-1.)
        try:ns['backward'](ctx,go,gz)
        except AtLaunch:pass
        else:raise AssertionError('original launch missing')
        args,grid=captured[-1]
        passed=args[3 if which=='do' else 4]
        assert passed is lazy and passed.is_neg()
        # T=K=V=1, Q=K=V=1: dq=dk=2*(dO+dZ), dv=dO.
        actual_raw=[2*(ctypes.c_float.from_address(args[3].data_ptr()).value+ctypes.c_float.from_address(args[4].data_ptr()).value),ctypes.c_float.from_address(args[3].data_ptr()).value]
        logical_ref=[2*(go.item()+gz.item()),go.item()]
        assert actual_raw!=logical_ref
        witnesses.append({'cotangent':which,'contiguous_preserves_identity_and_lazy_flag':True,'original_passes_lazy_pointer':True,'raw_value':raw,'logical_value':logical,'raw_pointer_dq_dv':actual_raw,'logical_reference_dq_dv':logical_ref,'scope':'Actual original host body intercepted at launch; scalar pointer-semantics counterexample, not observed GPU output'})
    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x.clone()
        @staticmethod
        def backward(ctx,g):observed.append(g.is_neg());return g
    x=torch.ones(1,requires_grad=True)
    torch.autograd.grad(Probe.apply(x),x,torch._neg_view(torch.ones(1)))
    assert observed==[True]
    torch.Tensor.contiguous=track_copy
    for which in ['do','dz']:
        reject('lazy_negative_'+which,lambda:m.ParallelBasedFunction.backward(ctx,torch._neg_view(do) if which=='do' else do,torch._neg_view(dz) if which=='dz' else dz))
    qe=torch.empty(0,1,1,1);ze=torch.empty(0,1,1)
    ectx=SimpleNamespace(saved_tensors=(qe,qe,qe),scale=1.)
    reject('empty_lazy_negative_do',lambda:m.ParallelBasedFunction.backward(ectx,torch._neg_view(qe),ze))
    reject('empty_lazy_negative_dz',lambda:m.ParallelBasedFunction.backward(ectx,qe,torch._neg_view(ze)))
    for name,go,gz in [('non_tensor',None,dz),('do_dtype',do.double(),dz),('dz_dtype',do,dz.half()),('do_shape',do[...,0],dz),('dz_shape',do,dz[...,None]),('sparse_layout',do.to_sparse(),dz)]:
        reject(name,lambda go=go,gz=gz:m.ParallelBasedFunction.backward(ctx,go,gz))
    torch.Tensor.is_cuda=property(lambda t:id(t)!=id(do))
    reject('cuda_residency_observation',lambda:m.ParallelBasedFunction.backward(ctx,do,dz))
    torch.Tensor.is_cuda=property(lambda _:True)
    torch.Tensor.device=property(lambda t:torch.device('cuda:1') if id(t)==id(do) else old_tensor_device.__get__(t))
    reject('device_mismatch_observation',lambda:m.ParallelBasedFunction.backward(ctx,do,dz))
    torch.Tensor.device=old_tensor_device
    torch.Tensor.is_conj=lambda t:id(t)==id(dz) or old_conj(t)
    reject('conjugate_flag_observation',lambda:m.ParallelBasedFunction.backward(ctx,do,dz))
    torch.Tensor.is_conj=old_conj
    q2=torch.ones(1,1,2,2);v2=q2.clone();go=v2.transpose(-1,-2);gz=torch.ones(1,1,2)
    reject('prepared_noncontiguous',lambda:m._check_prepared_cotangents(q2,q2,v2,go,gz))
    # Actual expanded views and copied ready tensors are admitted.
    go=torch.ones(1).expand_as(v2);gz=torch.ones(1).expand(1,1,2)
    m._check_backward_inputs(q2,q2,v2,go,gz)
    m._check_prepared_cotangents(q2,q2,v2,old_contiguous(go),old_contiguous(gz))
    # Storage queries below are synthetic controls for otherwise impossible
    # ordinary Tensor metadata; they exercise the exact production helper.
    def metadata(pointer=64,base=64,capacity=4,stride=1,size=1):
        return SimpleNamespace(stride=lambda:(stride,),numel=lambda:size,shape=(size,),data_ptr=lambda:pointer,untyped_storage=lambda:SimpleNamespace(data_ptr=lambda:base,nbytes=lambda:capacity))
    for name,x,limit in [('negative_stride',metadata(stride=-1),None),('unaligned',metadata(pointer=65,capacity=8),None),('below_storage',metadata(pointer=60),None),('short_storage',metadata(capacity=3),None),('strided_span',metadata(size=2,stride=2,capacity=8),None),('storage_end_signed64',metadata(pointer=2**63-4,base=2**63-4,capacity=8),None),('formed_pointer_signed64',metadata(pointer=2**63-4,base=2**63-4,capacity=4),1)]:
        reject('synthetic_'+name,lambda x=x,limit=limit:m._check_cotangent_storage(x,limit))
    m._check_cotangent_storage(metadata(size=4,stride=0))
finally:
    torch.Tensor.is_cuda=old_cuda;torch.cuda.device=old_device
    torch.Tensor.contiguous=old_contiguous;torch.Tensor.device=old_tensor_device;torch.Tensor.is_conj=old_conj
    m.parallel_rebased_bwd_kernel=old_launch
receipt={'source_sha256':sha(P),'original_sha256':sha(O),'script_sha256':sha(Path(__file__)),'torch_version':torch.__version__,'GPU_execution':False,'original_lazy_negative_witnesses':witnesses,'autograd_preserves_lazy_negative_cotangent':observed==[True],'rejections':rejected,'expanded_copy_and_zero_stride_admitted':True,'all_passed':len(rejected)==21 and len(witnesses)==2,'observation_limits':'CUDA residency, device mismatch and FP32 conjugate-flag controls use explicit metadata observation shims; storage controls use synthetic query records. Original/source pointer witnesses and lazy-negative flags are actual CPU Torch observations.'}
assert receipt['all_passed'],len(rejected)
(D/'backward-contract.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':True,'rejections':len(rejected),'original_witnesses':len(witnesses)}))
