"""Actual wrapper/CPU-interpreter controls and metadata rejection checks."""
import ast
import hashlib,json,sys
sys.dont_write_bytecode=True
import contextlib
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
os.environ['TRITON_INTERPRET']='1'
import torch
import triton
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'chunk_delta_fwd_static_dot.py'
BASE=HERE.parent/'chunk_delta_fwd/chunk_delta_fwd_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunk_delta_fwd/chunk_delta_fwd.py'
KERNEL='chunk_delta_rule_fwd_kernel_h'

def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

def floating(fn):
    result=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load':
            n.keywords=[kw for kw in n.keywords if kw.arg!='padding_option']
    for n in ast.walk(fn):
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            ts=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Name) and t.id.startswith('b_') for t in ts):result.append(ast.dump(n))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='store':result.append(ast.dump(n.args[1]))
    return sorted(result)

def source_checks():
    old=function(ORIGINAL,KERNEL);new=function(SOURCE,KERNEL)
    assert ast.dump(function(BASE,KERNEL))==ast.dump(new)
    before=function(BASE,'chunk_fwd_h_fn_repaired');after=function(SOURCE,'chunk_fwd_h_fn_static_dot')
    before.name=after.name
    before.body=before.body[1:];after.body=after.body[1:]
    guard=[n for n in after.body if isinstance(n,ast.If) and 'launched FP32 dots' in ast.unparse(n)]
    assert len(guard)==1
    after.body.remove(guard[0])
    assert ast.dump(before)==ast.dump(after)
    assert floating(old)==floating(new)
    assert all(ast.dump(x)==ast.dump(y) for x,y in zip(old.decorator_list,new.decorator_list))
    assert len(old.decorator_list)==len(new.decorator_list)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','float','resolve_neg','resolve_conj') for n in ast.walk(ast.parse(SOURCE.read_text())))
    return len(floating(old))

class MetaTensor:
    def __init__(self,shape,*,dtype=torch.float32,cuda=True,device='cuda:1',contiguous=True,negative=False,conjugate=False,pointer=4096,layout=torch.strided):
        self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.is_cuda=cuda;self.device=torch.device(device);self.layout=layout
        self.contig=contiguous;self.negative=negative;self.conjugate=conjugate;self.pointer=pointer
    def is_contiguous(self):return self.contig
    def is_neg(self):return self.negative
    def is_conj(self):return self.conjugate
    def data_ptr(self):return self.pointer
    def numel(self):return math.prod(self.shape)

def rejection_checks():
    def unexpected(*a,**k):raise AssertionError('rejection reached allocation/setup')
    namespace={'torch':SimpleNamespace(Tensor=MetaTensor,strided=torch.strided,float32=torch.float32,empty=unexpected,cuda=SimpleNamespace(device=unexpected)), 'triton':triton}
    functions=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name!=KERNEL]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),namespace)
    run=namespace['chunk_fwd_h_fn_static_dot'];count=0
    for index in range(5):
        for option in [dict(dtype=torch.float16),dict(cuda=False),dict(contiguous=False),dict(negative=True),dict(conjugate=True),dict(pointer=1),dict(pointer=2**63-4),dict(device='cuda:2'),dict(layout=torch.sparse_coo)]:
            args=[MetaTensor((1,1,17,16)),MetaTensor((1,1,17,16)),MetaTensor((1,1,17,5)),16,MetaTensor((1,1,16,5),pointer=65536),MetaTensor((1,1,16,5),pointer=131072)]
            pos=[0,1,2,4,5][index]
            args[pos]=MetaTensor(args[pos].shape,**option)
            try:run(*args)
            except (TypeError,ValueError):count+=1
            else:raise AssertionError((index,option))
    for BT in [True,False,0,-1,3,24,2**31]:
        try:run(MetaTensor((1,1,17,16)),MetaTensor((1,1,17,16)),MetaTensor((1,1,17,5)),BT)
        except (TypeError,ValueError):count+=1
        else:raise AssertionError(BT)
    for shape in [(1,1,17,0),(1,1,17,257),(65536,1,1,16),(1,1,2**30,16)]:
        B,H,T,K=shape
        try:run(MetaTensor(shape),MetaTensor(shape),MetaTensor((B,H,T,5)),16)
        except (TypeError,ValueError):count+=1
        else:raise AssertionError(shape)
    for v in [65536*64,2**31]:
        try:run(MetaTensor((1,1,17,16)),MetaTensor((1,1,17,16)),MetaTensor((1,1,17,v)),16)
        except (TypeError,ValueError):count+=1
        else:raise AssertionError(v)
    for final in [MetaTensor((1,1,16,5),pointer=4096),MetaTensor((1,1,16,5),pointer=65536)]:
        try:run(MetaTensor((1,1,17,16)),MetaTensor((1,1,17,16)),MetaTensor((1,1,17,5)),16,MetaTensor((1,1,16,5),pointer=65536),final)
        except ValueError:count+=1
        else:raise AssertionError('alias guard')
    return count

def reference(k,w,u,BT,seed):
    B,H,T,K=k.shape;V=u.shape[-1];BK=triton.next_power_of_2(K);BV=16 if BK>128 else 32 if BK>64 else 64;BC=min(BT,BV)
    state=torch.zeros(B,H,K,V) if seed is None else seed.clone()
    history=[];corrected=torch.empty_like(u)
    for c in range(triton.cdiv(T,BT)):
        history.append(state.clone());acc=torch.zeros_like(state)
        for b in range(triton.cdiv(BT,BC)):
            lo=c*BT+b*BC;hi=min(lo+BC,T)
            if lo>=T:continue
            values=u[:,:,lo:hi]-w[:,:,lo:hi]@state
            corrected[:,:,lo:hi]=values
            acc+=k[:,:,lo:hi].transpose(-1,-2)@values
        state+=acc
    h=torch.cat(history,dim=2) if history else torch.empty(B,H,0,V)
    return h,corrected,state

def interpreter_checks():
    spec=importlib.util.spec_from_file_location('delta_controls',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.chunk_delta_rule_fwd_kernel_h=m.chunk_delta_rule_fwd_kernel_h.fn
    events=[];current=['caller']
    @contextlib.contextmanager
    def owner(device):
        saved=current[0];current[0]=device;events.append(('enter',device))
        try:yield
        finally:current[0]=saved;events.append(('exit',saved))
    def cpu_tensor(t,shape,name,device=None):
        assert isinstance(t,torch.Tensor) and tuple(t.shape)==tuple(shape) and t.dtype==torch.float32
        assert t.is_contiguous() and not t.is_neg() and not t.is_conj()
        assert device is None or t.device==device
    m._tensor=cpu_tensor
    m.torch=SimpleNamespace(Tensor=torch.Tensor,strided=torch.strided,float32=torch.float32,empty=torch.empty,cuda=SimpleNamespace(device=owner))
    count=0
    for T,K,V,BT in [(17,9,3,16),(16,16,5,16),(17,17,19,16),(85,16,17,80),(33,129,5,17)]:
        k=torch.zeros(1,2,T,K)
        for t in range(T):k[:,:,t,t%K]=1
        w=k;u=(torch.arange(T*V).reshape(1,1,T,V)%3).float().expand(1,2,T,V).contiguous()
        for seed in [None,torch.full((1,2,K,V),2.)]:
            for wants_final in [False,True]:
                target=torch.empty(1,2,K,V) if wants_final else None
                snapshots=[x.clone() for x in [k,w,u]+([] if seed is None else [seed])]
                expected=reference(k,w,u,BT,seed)
                got=m.chunk_fwd_h_fn_static_dot(k,w,u,BT,seed,target)
                for a,b in zip(got,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
                if target is not None:torch.testing.assert_close(target,expected[2],rtol=0,atol=0)
                for a,b in zip([k,w,u]+([] if seed is None else [seed]),snapshots):assert torch.equal(a.view(torch.int32),b.view(torch.int32))
                assert got[0].is_contiguous() and got[1].is_contiguous() and current[0]=='caller';count+=1
    # Harmless singleton strides are admitted; original source pointers are used.
    dense=torch.ones(16,16);k=torch.as_strided(dense,(1,1,16,16),(0,0,16,1));u=torch.ones(1,1,16,3)
    got=m.chunk_fwd_h_fn_static_dot(k,k,u,16)
    expected=reference(k,k,u,16,None)
    for a,b in zip(got,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    count+=1
    # T=0 executes the original state load/store path, preserving signed zeros.
    for seed in [None,torch.full((1,1,9,3),-0.)]:
        k=torch.empty(1,1,0,9);u=torch.empty(1,1,0,3);target=torch.empty(1,1,9,3)
        got=m.chunk_fwd_h_fn_static_dot(k,k,u,16,seed,target)
        expect=torch.zeros_like(target) if seed is None else seed
        assert torch.equal(target.view(torch.int32),expect.view(torch.int32))
        assert all(x.numel()==0 for x in got);count+=1
    for shape in [(0,1,17,1,5),(1,0,17,1,5),(1,1,17,1,0)]:
        B,H,T,K,V=shape;k=torch.empty(B,H,T,K);u=torch.empty(B,H,T,V);target=torch.empty(B,H,K,V)
        got=m.chunk_fwd_h_fn_static_dot(k,k,u,1,None,target)
        assert all(x.numel()==0 for x in got);count+=1
    assert current[0]=='caller' and len(events)==2*count
    return count


def static_domain_checks():
    calls=[]
    class Capture:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):calls.append((grid,args,kwargs))
            return launch
    namespace={'torch':SimpleNamespace(Tensor=MetaTensor,strided=torch.strided,float32=torch.float32,
        empty=lambda shape,dtype,device:MetaTensor(shape,device=device,pointer=1048576),
        cuda=SimpleNamespace(device=lambda device:contextlib.nullcontext())),
        'triton':triton,KERNEL:Capture()}
    functions=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name!=KERNEL]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),namespace)
    run=namespace['chunk_fwd_h_fn_static_dot'];admit=reject=0
    cases=[(1,1,17,1,3,16,False),(1,1,17,8,3,16,False),(1,1,17,9,3,16,True),
           (1,1,17,16,3,1,False),(1,1,17,16,3,8,False),
           (1,1,0,1,3,16,False),(1,1,0,16,3,1,False),(1,1,0,9,3,16,True),
           (0,1,17,1,3,1,True),(1,0,17,1,3,1,True),(1,1,17,1,0,1,True),
           (1,1,85,9,3,80,True),(1,1,33,129,3,17,True)]
    for B,H,T,K,V,BT,accepted in cases:
        calls.clear()
        args=[MetaTensor((B,H,T,K)),MetaTensor((B,H,T,K)),MetaTensor((B,H,T,V)),BT]
        try:run(*args)
        except ValueError as error:
            assert not accepted and 'launched FP32 dots' in str(error)
            assert not calls;reject+=1
        else:
            assert accepted and bool(calls)==bool(B*H*V)
            if calls:
                cfg=calls[0][2]
                assert cfg['BK']>=16 and cfg['BC']>=16
            admit+=1
    return admit,reject

if __name__=='__main__':
    source=source_checks();old_rejections=rejection_checks()
    admissions,rejections=static_domain_checks();grids=interpreter_checks()
    report={'status':'PASS','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'floating_statements_preserved':source,'kernel_AST_identical':True,
        'wrapper_AST_diff':'exactly the documented static-dot guard plus name/docstring',
        'inherited_metadata_rejections':old_rejections,'new_domain_admissions':admissions,
        'new_domain_rejections':rejections,'CPU_interpreter_controls':grids,
        'GPU_execution':False,'scope':'Contract/boundary controls with exact integer inputs; no GPU or numerical accuracy experiment.'}
    out=HERE.parents[3]/'.veritile-cache/kernel-evidence/chunk_delta_fwd_static_dot/cpu-contract.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
