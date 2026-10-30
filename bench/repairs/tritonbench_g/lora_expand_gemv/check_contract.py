"""Actual checked host/kernel AST; CPU arithmetic is not native GPU conformance."""
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
import triton.language as real_tl
import triton.language.standard as standard

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'lora_expand_gemv_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/lora_expand_gemv/lora_expand_gemv.py'
spec=importlib.util.spec_from_file_location('ff_helpers',HERE.parent/'llama_ff_triton/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,Pointer,typedraw=helpers.function,helpers.raw,helpers.Pointer,helpers.typedraw
EVENTS=[]

class Tensor(helpers.Tensor):
    def tolist(self):EVENTS.append(('tolist',self));return self.data.tolist()
    def size(self,dim):return self.shape[dim]
    def squeeze(self,dim):
        t=Tensor(self.data.squeeze(dim),self.is_cuda,self.device.index or 0);EVENTS.append(('squeeze',self,t));return t
    def __eq__(self,other):return self.data==other
    def __ge__(self,other):return self.data>=other
    def __lt__(self,other):return self.data<other

def kernel_ast(path):
    node=function(path,'_bgmv_expand_kernel');node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    pid=[0,0]
    def offsets(p,mask):
        off=torch.as_tensor(p.offsets,dtype=torch.int64)
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.broadcast_to(torch.as_tensor(mask,dtype=torch.bool),off.shape)
        ix=p.tensor.storage_offset()+off[mask]
        assert torch.all(ix>=0) and torch.all(ix<typedraw(p.tensor).numel()),'active access outside caller backing'
        return off,mask,ix
    def load(p,mask=None,other=0):
        off,mask,ix=offsets(p,mask);out=torch.full_like(off,other,dtype=p.tensor.dtype);out[mask]=typedraw(p.tensor)[ix];return out
    def store(p,v,mask=None):
        off,mask,ix=offsets(p,mask);typedraw(p.tensor)[ix]=torch.broadcast_to(v,off.shape)[mask].to(p.tensor.dtype)
    tl=SimpleNamespace(program_id=lambda axis:torch.tensor(pid[axis]),arange=torch.arange,int64=torch.int64,
      cast=lambda v,dtype:torch.as_tensor(v,dtype=dtype),cdiv=lambda a,b:(a+b-1)//b,
      load=load,store=store,max_contiguous=lambda a,n:a,sum=lambda a,axis:a.sum(dim=axis,dtype=a.dtype))
    scope={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope['_bgmv_expand_kernel'],pid

def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    for key,path in [('original_sha256',ORIGINAL),('candidate_sha256',SOURCE),('quotation_sha256',HERE/'Kernels.lean')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[key]
    assert hashlib.sha256(ast.dump(function(SOURCE,'_bgmv_expand_kernel')).encode()).hexdigest()==pins['historical_kernel_ast_sha256']
    assert json.dumps(ast.dump(ast.parse(SOURCE.read_text()))) in (HERE/'Proofs/Direct/Compiler.lean').read_text()
    wrapper=function(SOURCE,'_bgmv_expand')
    calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'clone','contiguous','resolve_neg','copy_','index_copy_','zeros','empty'}
    profile=json.loads((HERE/'numerical_profile.json').read_text())
    assert triton.__version__==profile['triton_version']
    assert hashlib.sha256(inspect.getsource(standard.sum.fn).encode()).hexdigest()==profile['sum_sha256']
    assert hashlib.sha256(inspect.getsource(standard._pick_sum_dtype.fn).encode()).hexdigest()==profile['sum_selector_sha256']
    assert standard._pick_sum_dtype.fn(real_tl.float16,None) is None
    assert hashlib.sha256(inspect.getsource(triton.next_power_of_2.fn).encode()).hexdigest()==profile['next_power_of_two_sha256']
    assert all(k<=triton.next_power_of_2(k)<=4096 and (k%triton.next_power_of_2(k)!=0 or k==triton.next_power_of_2(k)) for k in range(1,4097))
    print('Historical kernel AST unchanged; installed Triton 3.7.1 half-reduction dtype selection pinned.')

def host_checks():
    torch.manual_seed(31)
    node=function(SOURCE,'_bgmv_expand');node.decorator_list=[]
    scope={'torch':torch,'triton':triton};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
    k,pid=kernel_ast(SOURCE);active=[2]
    @contextlib.contextmanager
    def device(d):
        saved=active[0];active[0]=d.index;EVENTS.append(('enter',d.index))
        try:yield
        finally:active[0]=saved;EVENTS.append(('exit',saved))
    class Launch:
        fail=False
        reverse=False
        def __getitem__(self,grid):
            def run(*args,**kw):
                EVENTS.append(('launch',grid,args,kw,active[0]));assert active[0]==args[0].device.index
                values=[Pointer(a.data) if isinstance(a,Tensor) else a for a in args]
                order=[(n,b) for b in range(grid[1]) for n in range(grid[0])]
                if self.reverse:order.reverse()
                for n,b in order:
                    pid[:]=[n,b];k(*values,**kw)
                    if self.fail:raise RuntimeError('injected partial launch')
            return run
    launch=Launch();scope['_bgmv_expand_kernel']=launch
    scope['torch']=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float16=torch.float16,float32=torch.float32,
      int32=torch.int32,int64=torch.int64,all=torch.all,cuda=SimpleNamespace(device=device))
    fn=scope['_bgmv_expand']
    def inputs(b,n,k,four=False,ids=None):
        x=Tensor(torch.randn(b+1,k)*.3);w=Tensor(torch.randn((3,1,n,k) if four else (3,n,k),dtype=torch.float16)*.3)
        y=Tensor(torch.randn(b+2,n+8));i=Tensor(torch.tensor(ids if ids is not None else [j%3 if j%3 else -1 for j in range(b)],dtype=torch.int32))
        return [x,w,y,i,True]
    good=[inputs(3,n,k,four) for n,k in ((1,1),(3,7),(5,32),(17,65),(2,129),(1,4096)) for four in (False,True)]
    good += [inputs(0,3,7),inputs(2,0,7),inputs(4,3,5,ids=[-1]*4),inputs(1,3,5,ids=[2])]
    for case in list(good[:4]):good.append([*case[:4],False])
    # Aligned offset views preserve their original backing, pointer and row pitch.
    case=inputs(2,3,5);case[0]=Tensor(torch.randn(40)[4:19].view(3,5));case[3]=Tensor(torch.tensor([0,-1],dtype=torch.int64));good.append(case)
    case=inputs(2,3,5,ids=[0,2]);shared=torch.randn(45)*.3
    case[0]=Tensor(shared[:15].view(3,5));case[1]=Tensor(shared.view(torch.float16)[:45].view(3,3,5));good.append(case)
    case=inputs(2,3,5,ids=[0,2]);case[2]=Tensor(torch.randn(80)[4:48].view(4,11));good.append(case)
    case=inputs(2,3,5,ids=[0,1]);case[3]=Tensor(case[3].data.long());good.append(case)
    case=inputs(2,3,5,ids=[0,1]);shared=torch.zeros(45,dtype=torch.float32)
    shared.view(torch.int32)[:2]=torch.tensor([0,1],dtype=torch.int32)
    case[0]=Tensor(shared[:15].view(3,5));case[3]=Tensor(shared.view(torch.int32)[:2]);good.append(case)
    for case_no,args in enumerate(good):
        launch.reverse=bool(case_no%2)
        x,w,y,i,add=args;offset=0;n=w.shape[-2];before=[raw(t.data).clone() for t in (x,w,y,i)];expected=y.data.clone();weights=w.data.squeeze(1) if w.ndim==4 else w.data
        for b,slot in enumerate(i.data.tolist()):
            if slot==-1:continue
            products=(x.data[b].half()[None,:]*weights[slot]).half()
            value=products.sum(1,dtype=torch.float16).float()
            if add:value=value+expected[b,offset:offset+n]
            expected[b,offset:offset+n]=value
        EVENTS.clear();result=fn(*args)
        assert result is None and active[0]==2
        torch.testing.assert_close(y.data,expected,rtol=2e-3,atol=2e-3,equal_nan=True)
        for j,t in enumerate((x,w,y,i)):
            if j!=2:assert torch.equal(raw(t.data),before[j])
        # Exact bytes outside active destinations, including whole backing tails.
        allowed=torch.zeros(raw(y.data).numel(),dtype=torch.bool)
        for b,slot in enumerate(i.data.tolist()):
            if slot!=-1:
                for c in range(offset,offset+n):allowed[4*(y.data.storage_offset()+b*y.shape[1]+c):4*(y.data.storage_offset()+b*y.shape[1]+c+1)]=True
        assert torch.equal(raw(y.data)[~allowed],before[2][~allowed])
        launches=[e for e in EVENTS if e[0]=='launch']
        if i.numel() and n:
            assert len(launches)==1
            _,grid,actual,kw,dev=launches[0]
            assert grid==(n,i.numel()) and actual[0] is x and actual[2] is y and actual[5] is i
            assert actual[1].untyped_storage()._cdata==w.untyped_storage()._cdata
            assert actual[6:]==(x.shape[1],1,n*x.shape[1],x.shape[1],1,y.shape[1],1)
        else:assert not any(e[0] in ('enter','launch','squeeze') for e in EVENTS)
    bad=[]
    for side in range(4):
        base=inputs(2,3,5)[side]
        vals=[None,Tensor(base.data,cuda=False),Tensor(base.data,device=1),Tensor(torch._neg_view(base.data)),Tensor(base.data.double())]
        if base.ndim>1:vals += [Tensor(base.data.transpose(-1,-2))]
        for value in vals:
            case=inputs(2,3,5);case[side]=value;bad.append(case)
    for values in ([-2,0],[3,0],[2**31-1,-1]):bad.append(inputs(2,3,5,ids=values))
    for pos,value in ((4,-1),(4,1),(4,None)):
        case=inputs(2,3,5);case[pos]=value;bad.append(case)
    bad += [inputs(2,3,0),inputs(1,1,4097)]
    case=inputs(2,3,5);case[3]=Tensor(torch.tensor([-1,-1,0],dtype=torch.int32)[::2]);bad.append(case)
    case=inputs(2,3,5);case[2]=Tensor(case[0].data);bad.append(case)
    case=inputs(2,3,5);case[1]=Tensor(torch.randn(3,2,3,5,dtype=torch.float16));bad.append(case)
    case=inputs(2,3,5);case[0]=Tensor(torch.randn(16)[1:].view(3,5));bad.append(case)
    case=inputs(2,0,5,ids=[9,0]);bad.append(case)
    for role in (1,3):
        case=inputs(2,3,5);read=case[role].data;y=case[2].data
        backing=torch.empty(raw(read).numel()+raw(y).numel()+32,dtype=torch.uint8)
        rv=backing[:raw(read).numel()].view(read.dtype).view(read.shape);rv.copy_(read)
        off=(raw(read).numel()+15)//16*16
        yv=backing[off:off+raw(y).numel()].view(y.dtype).view(y.shape);yv.copy_(y)
        case[role]=Tensor(rv);case[2]=Tensor(yv);bad.append(case)
    for number,case in enumerate(bad):
        EVENTS.clear();snap=[(t,raw(t.data).clone()) for t in case[:4] if isinstance(t,Tensor) and t.layout==torch.strided]
        try:fn(*case)
        except (ValueError,TypeError):pass
        else:raise AssertionError(f'invalid call {number} admitted')
        assert not any(e[0] in ('enter','launch','squeeze') for e in EVENTS)
        for t,saved in snap:assert torch.equal(raw(t.data),saved)
    case=inputs(2,3,5,ids=[0,1]);before=[raw(t.data).clone() for t in case[:4]];EVENTS.clear();launch.fail=True;launch.reverse=False
    try:fn(*case)
    except RuntimeError:pass
    else:raise AssertionError('failure swallowed')
    assert active[0]==2 and EVENTS[-1]==('exit',2)
    for j in (0,1,3):assert torch.equal(raw(case[j].data),before[j])
    changed=raw(case[2].data)!=before[2];allowed=torch.zeros_like(changed);allowed[0:4]=True
    assert not torch.any(changed&~allowed)
    print(f'{len(good)} complete direct cases, {len(bad)} rejection cases and partial-write failure frame passed.')
    print('Both add modes, sentinel and empty paths, K tails and rank4096, 3D/4D weights, metadata widths, direct identities and all untouched caller bytes checked. CPU AST execution only.')

if __name__=='__main__':source_binding();host_checks()
