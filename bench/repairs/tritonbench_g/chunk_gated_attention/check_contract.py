"""Actual-source AST and CPU execution controls for separate checked APIs."""
import ast
import contextlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
os.environ['TRITON_INTERPRET']='1'
import torch
import triton

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'chunk_gated_attention_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunk_gated_attention/chunk_gated_attention.py'
NAMES=['chunk_gated_abc_fwd_kernel_cum','chunk_gated_abc_fwd_kernel_h']

def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

def floats(fn):
    nodes=[]
    for n in ast.walk(fn):
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Name) and (t.id.startswith('b_') or t.id=='m_s') for t in targets):
                nodes.append(ast.dump(n))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='store':
            nodes.append(ast.dump(n.args[1]))
    return sorted(nodes)

def source_checks():
    count=0
    for name in NAMES:
        old=function(ORIGINAL,name);new=function(SOURCE,name)
        assert floats(old)==floats(new);count+=len(floats(old))
        assert len(old.decorator_list)==len(new.decorator_list)
        assert all(ast.dump(a)==ast.dump(b) for a,b in zip(old.decorator_list,new.decorator_list))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','resolve_conj','resolve_neg','float') for n in ast.walk(ast.parse(SOURCE.read_text())))
    return count

class MetaTensor:
    def __init__(self,shape,*,dtype=torch.float32,cuda=True,device='cuda:1',contiguous=True,negative=False,conjugate=False,pointer=4096):
        self.shape=shape;self.dtype=dtype;self.layout=torch.strided;self.is_cuda=cuda;self.device=torch.device(device)
        self.contig=contiguous;self.negative=negative;self.conjugate=conjugate;self.pointer=pointer
    def is_contiguous(self):return self.contig
    def is_neg(self):return self.negative
    def is_conj(self):return self.conjugate
    def data_ptr(self):return self.pointer
    def numel(self):
        import math
        return math.prod(self.shape)

def rejection_checks():
    def unexpected(*a,**k):raise AssertionError('admission reached allocation/setup')
    namespace={'torch':SimpleNamespace(Tensor=MetaTensor,strided=torch.strided,float32=torch.float32,empty=unexpected,cuda=SimpleNamespace(device=unexpected)),
               'triton':triton,'_MAX_GRID_X':2**31-1}
    fns=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name not in NAMES]
    exec(compile(ast.Module(body=fns,type_ignores=[]),str(SOURCE),'exec'),namespace)
    pre=namespace['fwd_pre_repaired'];inner=namespace['fwd_inner_repaired'];rejects=0
    for options in [dict(dtype=torch.float16),dict(cuda=False),dict(contiguous=False),dict(negative=True),dict(conjugate=True),dict(pointer=1),dict(pointer=2**64-4)]:
        try:pre(MetaTensor((1,1,17,17),**options),1,1,17,17,16)
        except (TypeError,ValueError):rejects+=1
        else:raise AssertionError('pre accepted bad metadata')
    for bt in [True,8,24,2048]:
        try:pre(MetaTensor((1,1,17,17)),1,1,17,17,bt)
        except (TypeError,ValueError):rejects+=1
        else:raise AssertionError('block guard')
    for dims in [(-1,1,17,17),(True,1,17,17),(1,1,17,2**62)]:
        try:pre(MetaTensor(dims),*dims,16)
        except (TypeError,ValueError):rejects+=1
        else:raise AssertionError('dimension/span guard')
    data=[MetaTensor((1,1,17,17)) for _ in range(4)]
    for index in range(4):
        for options in [dict(dtype=torch.float16),dict(cuda=False),dict(contiguous=False),dict(negative=True),dict(conjugate=True),dict(pointer=1),dict(pointer=2**64-4)]:
            args=data.copy();args[index]=MetaTensor((1,1,17,17),**options)
            try:inner(*args,1,1,17,17,17,16,16,16)
            except (TypeError,ValueError):rejects+=1
            else:raise AssertionError('inner accepted bad metadata')
    for flag in [1,None,'key']:
        try:inner(*data,1,1,17,17,17,16,16,16,gatek=flag)
        except TypeError:rejects+=1
        else:raise AssertionError('gate flag')
    return rejects

class CumulativeLaunch:
    def __init__(self,kernel,bs):self.kernel=kernel;self.bs=bs
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            return self.kernel[grid({'BS':self.bs})](*args,BS=self.bs,**kwargs)
        return invoke

def interpreter_checks():
    spec=importlib.util.spec_from_file_location('gated_controls',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cum=module.chunk_gated_abc_fwd_kernel_cum.fn
    events=[];current=['caller']
    @contextlib.contextmanager
    def owner(device):
        saved=current[0];current[0]=device;events.append(('enter',device))
        try:yield
        finally:current[0]=saved;events.append(('exit',saved))
    # Only CUDA admission/device context are replaced. Original host allocation,
    # padding, views, final copy-back and actual kernel bodies execute on CPU.
    def cpu_tensor(tensor,shape,name,device=None):
        assert isinstance(tensor,torch.Tensor) and tensor.shape==shape and tensor.dtype==torch.float32
        assert tensor.is_contiguous() and not tensor.is_neg() and not tensor.is_conj()
        assert device is None or tensor.device==device
    module._tensor=cpu_tensor
    module.torch=SimpleNamespace(**{name:getattr(torch,name) for name in ['Tensor','strided','float32','empty','zeros']},cuda=SimpleNamespace(device=owner))
    controls=0
    for T,S,BT in [(16,64,16),(17,17,16),(33,65,32)]:
        B,H=1,2;g=torch.ones(B,H,T,S)
        for bs in [16,32,64]:
            module.chunk_gated_abc_fwd_kernel_cum=CumulativeLaunch(cum,bs)
            out=module.fwd_pre_repaired(g,B,H,T,S,BT)
            expected=torch.tensor([t%BT+1 for t in range(T)],dtype=torch.float32)[None,None,:,None].expand_as(out)
            torch.testing.assert_close(out,expected,rtol=0,atol=0);assert out.is_contiguous();controls+=1
    for T,K,V,BT,BK,BV in [(16,16,16,16,16,16),(17,17,19,16,16,16),(33,65,17,32,64,16)]:
        B,H=1,2;NT=triton.cdiv(T,BT)
        q=torch.ones(B,H,T,K);k=q;v=torch.full((B,H,T,V),3.)
        for gatek in [False,True]:
            gate=torch.zeros(B,H,T,K if gatek else V)
            for seed in [None,torch.full((B,H,K,V),2.)]:
                target=torch.empty(B,H,K,V);pointer=target.data_ptr()
                history=module.fwd_inner_repaired(q,k,v,gate,B,H,T,K,V,BT,BK,BV,gatek,seed,target)
                expected=torch.cat([torch.full((B,H,K,V),(0. if seed is None else 2.)+3*min(c*BT,T)) for c in range(NT)],dim=2)
                torch.testing.assert_close(history,expected,rtol=0,atol=0)
                torch.testing.assert_close(target,torch.full_like(target,(0. if seed is None else 2.)+3*T),rtol=0,atol=0)
                assert target.data_ptr()==pointer and history.is_contiguous();controls+=1
    # Fully aligned inputs are reused. Seed/final alias remains safe.
    seed=torch.full((1,1,16,16),5.);q=torch.ones(1,1,16,16);v=torch.full_like(q,2.);g=torch.zeros_like(q)
    assert module._materialize(q,q.shape) is q
    out=module.fwd_inner_repaired(q,q,v,g,1,1,16,16,16,16,16,16,h0=seed,ht=seed)
    torch.testing.assert_close(out,torch.full_like(out,5.),rtol=0,atol=0)
    torch.testing.assert_close(seed,torch.full_like(seed,37.),rtol=0,atol=0);controls+=1
    # Final target aliases an operand; the source must finish every read first.
    out=module.fwd_inner_repaired(q,q,v,g,1,1,16,16,16,16,16,16,ht=v)
    torch.testing.assert_close(out,torch.zeros_like(out),rtol=0,atol=0)
    torch.testing.assert_close(v,torch.full_like(v,32.),rtol=0,atol=0);controls+=1
    # Empty time preserves the actual zero-plus-negative-zero operation.
    q=torch.empty(1,1,0,16);g=torch.empty(1,1,0,16);seed=torch.full((1,1,16,16),-0.);target=torch.ones_like(seed)
    out=module.fwd_inner_repaired(q,q,q,g,1,1,0,16,16,16,16,16,h0=seed,ht=target)
    assert out.shape==(1,1,0,16) and not torch.signbit(target).any();controls+=1
    assert current[0]=='caller' and all(events[i][0]=='enter' and events[i+1][0]=='exit' for i in range(0,len(events),2))
    return controls

if __name__=='__main__':
    statements=source_checks();rejects=rejection_checks();controls=interpreter_checks()
    print(f'PASS: {statements} preserved floating assignment/store ASTs and nine autotune configurations; {rejects} rejects; {controls} exact CPU controls')
    print(f'PASS: separate pre/inner, all BS configurations, both gate branches, seed/final aliases, independent supplied gates, tails, direct aligned operands, empty signed-zero behavior, device restoration; torch={torch.__version__}; Triton={triton.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')
