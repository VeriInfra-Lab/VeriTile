"""Preserved arithmetic AST, checked host branches, and exact CPU controls."""
import ast
import sys
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
SOURCE=HERE/'chunk_gla_simple_min_dot_tile.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunk_gla_simple/chunk_gla_simple.py'
NAME='chunk_simple_gla_fwd_kernel_o_repaired'

def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

def arithmetic(fn):
    result=[]
    for n in ast.walk(fn):
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Name) and t.id in ('b_o','b_s','m_s') for t in targets):
                result.append(ast.dump(n))
    return result

def source_checks():
    old=node(ORIGINAL,NAME.removesuffix('_repaired'));new=node(SOURCE,NAME)
    assert arithmetic(old)==arithmetic(new)
    assert len(old.decorator_list)==len(new.decorator_list)
    assert all(ast.dump(a)==ast.dump(b) for a,b in zip(old.decorator_list,new.decorator_list))
    loads=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
    assert len(loads)==5
    assert all(any(k.arg=='other' and isinstance(k.value,ast.Constant) and k.value.value==0 for k in n.keywords) for n in loads)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','resolve_neg','copy_') for n in ast.walk(ast.parse(SOURCE.read_text())))
    oldstore=next(n for n in ast.walk(old) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='store')
    newstore=next(n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='store')
    assert ast.dump(oldstore.args[1])==ast.dump(newstore.args[1])
    assert not any(isinstance(n,ast.Constant) and n.value==17 for n in ast.walk(new))

class Tensor:
    def __init__(self,data,*,shape=None,cuda=True,device='cuda:1',contiguous=None,negative=False,conjugate=False,pointer=None):
        self.data=data;self.shape=tuple(data.shape) if shape is None else tuple(shape);self.ndim=len(self.shape)
        self.dtype=data.dtype;self.layout=data.layout;self.is_cuda=cuda;self.device=torch.device(device)
        self.contig=data.is_contiguous() if contiguous is None else contiguous
        self.negative=negative;self.conjugate=conjugate;self.pointer=pointer
    def is_contiguous(self):return self.contig
    def is_neg(self):return self.negative or self.data.is_neg()
    def is_conj(self):return self.conjugate or self.data.is_conj()
    def data_ptr(self):return self.data.data_ptr() if self.pointer is None else self.pointer
    def numel(self):return math.prod(self.shape)

def host_checks():
    events=[];current=[torch.device('cuda:0')];inject={}
    @contextlib.contextmanager
    def device_context(d):
        before=current[0];current[0]=d;events.append(('enter',d))
        try:yield
        finally:current[0]=before;events.append(('exit',before))
    class Launch:
        def __getitem__(self,grid):
            def invoke(*args,**kwargs):
                assert current[0]==torch.device('cuda:1')
                events.append(('launch',grid,args,kwargs))
                if inject.get('setup'):raise RuntimeError('setup')
            return invoke
    def empty_like(v):
        if inject.get('allocation'):raise RuntimeError('allocation')
        out=Tensor(torch.empty_like(v.data),shape=v.shape,device=v.device,pointer=inject.get('pointer'))
        events.append(('allocate',out));return out
    namespace={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=device_context)), 'triton':triton,'math':math,NAME:Launch()}
    fns=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name!=NAME]
    exec(compile(ast.Module(body=fns,type_ignores=[]),str(SOURCE),'exec'),namespace)
    wrapper=namespace['chunk_fwd_o_fn_repaired']
    def arguments(B=1,H=2,T=17,K=17,V=17,BT=16):
        NT=triton.cdiv(T,BT)
        shapes=[(B,H,NT*K,V),(B,H,T,K),(B,H,T,K),(B,H,T,V),(B,H,T)]
        return [Tensor(torch.ones(1),shape=shape) for shape in shapes]+[BT,1.]
    good=arguments();bad=[]
    for index in range(5):
        for options in [dict(cuda=False),dict(device='cuda:2'),dict(contiguous=False),dict(negative=True),dict(conjugate=True),dict(pointer=1),dict(pointer=2**64-4)]:
            case=good.copy();case[index]=Tensor(torch.ones(1),shape=case[index].shape,**options);bad.append(case)
        case=good.copy();case[index]=Tensor(torch.ones(1).half(),shape=case[index].shape);bad.append(case)
        case=good.copy();case[index]=object();bad.append(case)
        case=good.copy();case[index]=Tensor(torch.ones(1));bad.append(case)
    for bt in [True,15,24,2048,16.]:case=good.copy();case[5]=bt;bad.append(case)
    for scale in [True,None,object(),float('inf'),float('nan'),1e39,10**500]:case=good.copy();case[6]=scale;bad.append(case)
    bad.extend([arguments(H=65536),arguments(T=65535*16+1),arguments(V=64*2**31),arguments(K=2**62)])
    for case in bad:
        before=len(events)
        try:wrapper(*case)
        except (TypeError,ValueError):pass
        else:raise AssertionError(('expected rejection',case))
        assert len(events)==before
    admits=0
    for K in [0,1,17,65]:
        case=arguments(K=K);before=len(events);out=wrapper(*case)
        assert [e[0] for e in events[before:]]==['allocate','enter','launch','exit']
        assert out is events[before][1]
        launch=events[before+2];args=launch[2]
        assert all(args[i] is case[j] for i,j in enumerate([1,2,3,0,4]))
        assert args[6:12]==(17*K,K,17*17,17,2*K*17,17)
        assert launch[3]['BK']==max(16,min(triton.next_power_of_2(max(K,1)),64))
        assert current[0]==torch.device('cuda:0');admits+=1
    # Read-only aliases, a storage-offset input, and noncanonical singleton strides.
    qdata=torch.arange(16*16+8,dtype=torch.float32)[8:].view(1,1,16,16)
    qdata=qdata.as_strided(qdata.shape,(777,999,16,1))
    case=arguments(H=1,T=16,K=16,V=16)
    case[1]=case[2]=Tensor(qdata);case[-1]=-0.
    before=len(events);wrapper(*case)
    assert events[before+2][2][0] is events[before+2][2][1]
    assert math.copysign(1.,events[before+2][2][12])==-1.;admits+=1
    for dims in [dict(B=0),dict(H=0),dict(T=0),dict(V=0)]:
        case=arguments(**dims);before=len(events);out=wrapper(*case)
        assert [e[0] for e in events[before:]]==['allocate'];admits+=1
    for pointer in [1,2**64-4]:
        inject['pointer']=pointer;before=len(events)
        try:wrapper(*good)
        except ValueError:pass
        else:raise AssertionError('output pointer guard')
        assert [e[0] for e in events[before:]]==['allocate'];inject.clear()
    for failure in ['allocation','setup']:
        inject[failure]=True
        try:wrapper(*good)
        except RuntimeError:pass
        else:raise AssertionError('runtime failure')
        inject.clear();assert current[0]==torch.device('cuda:0')
    return len(bad)+2,admits

def interpreter_checks():
    spec=importlib.util.spec_from_file_location('simple_controls',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    kernel=getattr(module,NAME).fn;count=0
    for T,K,V,BT in [(1,1,1,16),(1,0,1,16),(17,0,16,16),(1,8,1,16),(1,16,16,16),(17,17,19,16),(33,65,65,32),(17,129,17,16)]:
        B,H=1,2;NT=triton.cdiv(T,BT);BK=max(16,min(triton.next_power_of_2(max(K,1)),64));BV=min(triton.next_power_of_2(V),64)
        q=torch.ones(B,H,T,K);k=q;v=torch.ones(B,H,T,V);g=torch.zeros(B,H,T)
        history=torch.empty(B,H,NT*K,V)
        for chunk in range(NT):history[:,:,chunk*K:(chunk+1)*K,:]=chunk+1
        for scale in [1.,-.5]:
            out=torch.empty_like(v)
            kernel[(triton.cdiv(V,BV),NT,B*H)](q,k,v,history,g,out,T*K,K,T*V,V,NT*K*V,V,scale,T,K,V,BT,BK,BV)
            expected=torch.tensor([scale*K*((t//BT+1)+(t%BT+1)) for t in range(T)],dtype=torch.float32)[None,None,:,None].expand_as(out)
            torch.testing.assert_close(out,expected,atol=0,rtol=0);count+=1
    return count

if __name__=='__main__':
    source_checks();rejected,admitted=host_checks();controls=interpreter_checks()
    print(f'PASS: preserved arithmetic/casts/config AST; five explicit zero loads; {rejected} rejects; {admitted} admits; {controls} exact CPU controls')
    print(f'PASS: direct aliases/offset/singleton strides, allocation/setup/empty paths, output identity, canonical stride arguments, owner-device restoration; torch={torch.__version__}; Triton={triton.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')
