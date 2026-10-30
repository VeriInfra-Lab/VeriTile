"""Exact source AST, checked host control, and CPU interpreter controls."""
import ast
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
SOURCE=HERE/'chunk_gla_fwd_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunk_gla_fwd/chunk_gla_fwd.py'
NAMES=['chunk_gla_fwd_A_kernel_intra_sub_inter','chunk_gla_fwd_A_kernel_intra_sub_intra',
       'chunk_gla_fwd_A_kernel_intra_sub_intra_split','chunk_gla_fwd_A_kernel_intra_sub_intra_merge',
       'chunk_gla_fwd_kernel_o']
def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
class NormalizeLoads(ast.NodeTransformer):
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl':
            if n.func.attr in ('max_contiguous','multiple_of'):return n.args[0]
            if n.func.attr=='load':n.keywords=[k for k in n.keywords if k.arg!='padding_option']
        return n

def source_checks():
    for name in NAMES:
        assert ast.dump(NormalizeLoads().visit(node(ORIGINAL,name)))==ast.dump(NormalizeLoads().visit(node(SOURCE,name)))
    tree=ast.parse(SOURCE.read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','resolve_neg','copy_') for n in ast.walk(tree))
    for name in NAMES:
        for n in ast.walk(node(SOURCE,name)):
            if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):
                assert n.value!=17 # The native scale tag is -17, absent from all source literals.
    assert triton.language.core.TRITON_MAX_TENSOR_NUMEL==2**20

class Tensor:
    events=[]
    inject={}
    def __init__(self,data,*,shape=None,cuda=True,device='cuda:1',contiguous=None,negative=False,conjugate=False,pointer=None):
        self.data=data;self.shape=tuple(data.shape) if shape is None else tuple(shape);self.ndim=len(self.shape)
        self.dtype=data.dtype;self.layout=data.layout;self.is_cuda=cuda;self.device=torch.device(device)
        self.contig=data.is_contiguous() if contiguous is None else contiguous
        self.negative=negative;self.conjugate=conjugate;self.pointer=pointer
    def is_contiguous(self):return self.contig
    def is_neg(self):return self.negative or self.data.is_neg()
    def is_conj(self):return self.conjugate or self.data.is_conj()
    def data_ptr(self):return self.data.data_ptr() if self.pointer is None else self.pointer
    def new_zeros(self,shape,dtype):
        if self.inject.get('fail'):raise RuntimeError('allocation')
        out=Tensor(torch.zeros(shape,dtype=dtype),device=self.device,pointer=self.inject.get('pointer'))
        self.events.append(('allocate',out));return out

def host_checks(module):
    events=Tensor.events;current=[torch.device('cuda:0')];setup_fail=[False]
    @contextlib.contextmanager
    def device_context(d):
        before=current[0];current[0]=d;events.append(('enter',d))
        try:yield
        finally:current[0]=before;events.append(('exit',before))
    class Launch:
        def __init__(self,name):self.name=name
        def __getitem__(self,grid):
            def invoke(*args,**kwargs):
                assert current[0]==torch.device('cuda:1')
                if setup_fail[0]:raise RuntimeError('setup')
                events.append(('launch',self.name))
            return invoke
    def empty_like(v):
        out=Tensor(torch.empty_like(v.data),device=v.device,pointer=Tensor.inject.get('pointer'))
        events.append(('allocate',out));return out
    namespace={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=device_context)), 'triton':triton,'math':math}
    for name in NAMES:namespace[name]=Launch(name)
    fns=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name not in NAMES]
    exec(compile(ast.Module(body=fns,type_ignores=[]),str(SOURCE),'exec'),namespace)
    prod=namespace['chunk_fwd_intra_gated_gk_fn'];cons=namespace['chunk_fwd_o_gated_gk_fn']
    data=torch.ones(1,2,17,17);x=Tensor(data);bad=[]
    for arg in range(3):
        for value in [object(),Tensor(data.half()),Tensor(data,cuda=False),Tensor(data,device='cuda:2'),
                      Tensor(data,negative=True),Tensor(data,conjugate=True),Tensor(data,contiguous=False),
                      Tensor(data,pointer=1),Tensor(data,pointer=2**64-4)]:
            args=[x,x,x,1.,16];args[arg]=value;bad.append(args)
    for bt in [True,15,24,2048,16.]:bad.append([x,x,x,1.,bt])
    for scale in [True,None,object(),float('inf'),float('nan'),1e39,10**500]:bad.append([x,x,x,scale,16])
    bad.extend([[Tensor(torch.ones(1,2,17)),x,x,1.,16],[x,Tensor(torch.ones(1,2,16,17)),x,1.,16]])
    # Checked signed arithmetic and CUDA y/z axes without allocating huge tensors.
    for shape in [(1,65536,1,1),(1,1,2**31,1),(1,1,2**20,257)]:
        a=Tensor(data,shape=shape);bad.append([a,a,a,1.,16])
    for args in bad:
        before=len(events)
        try:prod(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError(('expected rejection',args))
        assert len(events)==before
    admitted=0
    for K in [1,17,256,257]:
        q=Tensor(torch.ones(1,2,17,K));before=len(events)
        out=prod(q,q,q,-0.,16)
        kinds=[x[0] for x in events[before:]]
        assert kinds==(['allocate','enter','launch','launch','exit'] if K<=256 else
                       ['allocate','enter','launch','allocate','launch','launch','exit'])
        assert out is next(e[1] for e in events[before:] if e[0]=='allocate')
        assert current[0]==torch.device('cuda:0');admitted+=1
    q=Tensor(torch.ones(1,2,17,17));v=Tensor(torch.ones(1,2,17,19));g=q
    a=Tensor(torch.ones(1,2,17,16));h=Tensor(torch.ones(1,2,34,19))
    out=cons(q,v,g,a,h,16,1.);assert out.shape==v.shape;admitted+=1
    # Standalone consumer attention/history are independent read-only inputs.
    for which in ['A','h']:
        before=len(events)
        try:cons(q,v,g,Tensor(torch.ones(1)) if which=='A' else a,Tensor(torch.ones(1)) if which=='h' else h,16,1.)
        except ValueError:pass
        else:raise AssertionError('shape guard')
        assert len(events)==before
    for shape in [(0,2,17,17),(1,2,0,17)]:
        z=Tensor(torch.empty(shape));before=len(events);out=prod(z,z,z,1.,16)
        assert [e[0] for e in events[before:]]==['allocate'];admitted+=1
    vv=Tensor(torch.empty(1,2,17,0));hh=Tensor(torch.empty(1,2,34,0));before=len(events)
    cons(q,vv,g,a,hh,16,1.);assert [e[0] for e in events[before:]]==['allocate'];admitted+=1
    for pointer in [1,2**64-4]:
        Tensor.inject['pointer']=pointer;before=len(events)
        try:prod(q,q,q,1.,16)
        except ValueError:pass
        else:raise AssertionError('output pointer')
        assert [e[0] for e in events[before:]]==['allocate'];Tensor.inject.clear()
    setup_fail[0]=True
    try:prod(q,q,q,1.,16)
    except RuntimeError:pass
    else:raise AssertionError('setup error')
    assert current[0]==torch.device('cuda:0')
    return len(bad)+4,admitted

def interpreter_checks(module):
    kernels={name:getattr(module,name).fn for name in NAMES}
    count=0
    for T,K,BT,V in [(1,1,16,16),(17,17,16,17),(33,31,32,19),(17,256,16,16),(17,257,16,17)]:
        B,H=1,2;BC=16;NT=triton.cdiv(T,BT);NC=BT//16
        q=torch.ones(B,H,T,K);k=q.clone();g=torch.zeros_like(q);A=torch.zeros(B,H,T,BT)
        kernels[NAMES[0]][(NT,NC*NC,B*H)](q,k,g,A,T*K,K,1.,T,K,BT,BC,min(64,triton.next_power_of_2(max(16,K))),NC)
        if K<=256:
            kernels[NAMES[1]][(NT,NC,B*H)](q,k,g,A,T*K,K,1.,T,K,BT,BC,triton.next_power_of_2(K))
        else:
            NK=triton.cdiv(K,128);scratch=torch.zeros(NK,B,H,T,BC)
            kernels[NAMES[2]][(NK,NT*NC,B*H)](q,k,g,scratch,T*K,K,1.,T,K,BT,BC,128,NC)
            kernels[NAMES[3]][(NT,NC,B*H)](scratch,A,T,BT,BC,NK)
        expected=torch.zeros_like(A)
        for t in range(T):expected[:,:,t,:t%BT+1]=K
        torch.testing.assert_close(A,expected,rtol=0,atol=0)
        v=torch.ones(B,H,T,V);history=torch.ones(B,H,NT*K,V);out=torch.empty_like(v)
        BK=min(32,triton.next_power_of_2(max(16,K)));BV=min(32,triton.next_power_of_2(max(1,V)))
        kernels[NAMES[4]][(triton.cdiv(V,BV),NT,B*H)](q,v,g,history,out,A,T*K,K,T*V,V,NT*K*V,V,1.,T,K,V,BT,BK,BV)
        expected=torch.tensor([K*(2+t%BT) for t in range(T)]).float()[None,None,:,None].expand_as(out)
        torch.testing.assert_close(out,expected,rtol=0,atol=0);count+=1
        A.fill_(3.)
        kernels[NAMES[4]][(triton.cdiv(V,BV),NT,B*H)](q,v,g,history,out,A,T*K,K,T*V,V,NT*K*V,V,1.,T,K,V,BT,BK,BV)
        expected=torch.tensor([K+3*(1+t%BT) for t in range(T)]).float()[None,None,:,None].expand_as(out)
        torch.testing.assert_close(out,expected,rtol=0,atol=0);count+=1
    return count

if __name__=='__main__':
    source_checks()
    spec=importlib.util.spec_from_file_location('gla_fwd_checked',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rejected,admitted=host_checks(module);controls=interpreter_checks(module)
    print(f'PASS: five preserved kernel ASTs modulo zero padding and removal of unchecked alignment hints; {rejected} rejects; {admitted} admits; {controls} exact CPU controls')
    print(f'PASS: allocation order, returned identity, owner-device restoration, both K branches, standalone consumer; torch={torch.__version__}; Triton={triton.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')
