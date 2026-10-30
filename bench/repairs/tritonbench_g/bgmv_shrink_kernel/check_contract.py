"""Checked host and actual kernel AST; CPU operations are not GPU conformance."""
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
import triton.language as real_tl
import triton.language.standard as standard

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'bgmv_shrink_kernel_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/bgmv_shrink_kernel/bgmv_shrink_kernel.py'
spec=importlib.util.spec_from_file_location('bgmv_helpers',HERE.parent/'bgmv_expand_slice/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,Pointer,typedraw=helpers.function,helpers.raw,helpers.Pointer,helpers.typedraw
EVENTS=helpers.EVENTS
Tensor=helpers.Tensor

def kernel_ast(path):
    node=function(path,'_bgmv_shrink_kernel');node.decorator_list=[]
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
    def atomic_add(p,v,mask=None):
        off,mask,ix=offsets(p,mask);typedraw(p.tensor)[ix]+=torch.broadcast_to(v,off.shape)[mask].to(p.tensor.dtype)
    tl=SimpleNamespace(program_id=lambda axis:torch.tensor(pid[axis]),arange=torch.arange,int64=torch.int64,float32=torch.float32,
      cast=lambda v,dtype:torch.as_tensor(v,dtype=dtype),zeros=torch.zeros,max_contiguous=lambda x,n:x,
      load=load,store=store,sum=lambda a,axis:a.sum(dim=axis,dtype=a.dtype),atomic_add=atomic_add)
    scope={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope['_bgmv_shrink_kernel'],pid

def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    for key,path in [('original_sha256',ORIGINAL),('candidate_sha256',SOURCE),('quotation_sha256',HERE/'Kernels.lean')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[key]
    assert hashlib.sha256(ast.dump(function(SOURCE,'_bgmv_shrink_kernel')).encode()).hexdigest()==pins['historical_kernel_ast_sha256']
    assert json.dumps(ast.dump(ast.parse(SOURCE.read_text()))) in (HERE/'Proofs/Direct/NativeGraph.lean').read_text()
    wrapper=function(SOURCE,'_bgmv_shrink')
    calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'clone','contiguous','resolve_neg','copy_','index_copy_','zeros','empty'}
    profile=json.loads((HERE/'numerical_profile.json').read_text())
    assert triton.__version__==profile['triton_version']
    assert hashlib.sha256(inspect.getsource(standard.sum.fn).encode()).hexdigest()==profile['sum_sha256']
    assert hashlib.sha256(inspect.getsource(standard._pick_sum_dtype.fn).encode()).hexdigest()==profile['sum_selector_sha256']
    assert standard._pick_sum_dtype.fn(real_tl.float16,None) is None
    assert standard._pick_sum_dtype.fn(real_tl.bfloat16,None) is None
    print('Historical numerical kernel AST unchanged; Triton 3.7.1 FP16/BF16 reduction dtype selection pinned.')

def host_checks():
    torch.manual_seed(32)
    node=function(SOURCE,'_bgmv_shrink');node.decorator_list=[]
    scope={'torch':torch,'triton':triton,'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
    kernel,pid=kernel_ast(SOURCE);active=[2]
    @contextlib.contextmanager
    def device(d):
        saved=active[0];active[0]=d.index;EVENTS.append(('enter',d.index))
        try:yield
        finally:active[0]=saved;EVENTS.append(('exit',saved))
    class Launch:
        fail=False
        def __getitem__(self,grid):
            def run(*args,**kw):
                EVENTS.append(('launch',grid,args,kw,active[0]));assert active[0]==args[0].device.index
                values=[Pointer(a.data) if isinstance(a,Tensor) else a for a in args]
                # Different split orders are legal; the source atomics stay separate.
                for b in range(grid[1]):
                    order=list(range(64)) if b%2==0 else list(reversed(range(64)))
                    for split in order:
                        pid[:]=[split,b];kernel(*values,**kw)
                        if self.fail:raise RuntimeError('injected partial launch')
            return run
    launch=Launch();scope['_bgmv_shrink_kernel']=launch
    scope['torch']=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float16=torch.float16,bfloat16=torch.bfloat16,
      float32=torch.float32,int32=torch.int32,int64=torch.int64,all=torch.all,cuda=SimpleNamespace(device=device))
    fn=scope['_bgmv_shrink']
    def values(shape,dtype):return ((torch.randint(0,3,shape)-1)/16).to(dtype)
    def inputs(b,n,k,dtype=torch.float16,four=False,ids=None,scale=1.0):
        x=Tensor(values((b+1,k),dtype));w=Tensor(values((3,1,n,k) if four else (3,n,k),dtype))
        y=Tensor(torch.randn(b+2,n+5));i=Tensor(torch.tensor(ids if ids is not None else [j%3 if j%3 else -1 for j in range(b)],dtype=torch.int32))
        return [x,w,y,i,scale]
    good=[inputs(3,n,k,dtype,four=k%2==0,scale=.5) for dtype in (torch.float16,torch.bfloat16)
          for n,k in ((1,0),(3,7),(5,255),(2,256),(3,257),(1,16385))]
    good += [inputs(0,3,7),inputs(2,0,7),inputs(3,3,7,ids=[-1]*3)]
    case=inputs(2,3,5,ids=[0,2]);case[0]=Tensor(values((40,),torch.float16)[8:23].view(3,5));case[3]=Tensor(torch.tensor([0,2],dtype=torch.int64));good.append(case)
    case=inputs(2,3,5,ids=[0,2]);shared=values((50,),torch.float16);case[0]=Tensor(shared[:15].view(3,5));case[1]=Tensor(shared[:45].view(3,3,5));good.append(case)
    case=inputs(2,3,5,ids=[0,2]);case[2]=Tensor(torch.randn(80)[4:36].view(4,8));good.append(case)
    for args in good:
        x,w,y,i,scale=args;before=[raw(t.data).clone() for t in (x,w,y,i)];expected=y.data.clone();weights=w.data.squeeze(1) if w.ndim==4 else w.data
        for b,slot in enumerate(i.data.tolist()):
            if slot!=-1:expected[b,:weights.shape[1]]+=scale*(x.data[b].float()[None,:]*weights[slot].float()).sum(1)
        EVENTS.clear();result=fn(*args)
        assert result is None and active[0]==2
        torch.testing.assert_close(y.data,expected,rtol=2e-5,atol=2e-5,equal_nan=True)
        for j,t in enumerate((x,w,y,i)):
            if j!=2:assert torch.equal(raw(t.data),before[j])
        allowed=torch.zeros(raw(y.data).numel(),dtype=torch.bool)
        for b,slot in enumerate(i.data.tolist()):
            if slot!=-1:
                start=4*(y.data.storage_offset()+b*y.shape[1]);allowed[start:start+4*weights.shape[1]]=True
        assert torch.equal(raw(y.data)[~allowed],before[2][~allowed])
        launches=[e for e in EVENTS if e[0]=='launch']
        if i.numel() and weights.shape[1]:
            assert len(launches)==1
            _,grid,actual,kw,dev=launches[0];n,k=weights.shape[-2:]
            assert grid==(64,i.numel()) and actual[0] is x and actual[2] is y and actual[5] is i
            assert actual[1].untyped_storage()._cdata==w.untyped_storage()._cdata
            assert actual[7:]==(k,1,n*k,k,1,y.shape[1],1)
            assert kw['BLOCK_K']==256 and kw['SPLIT_K']==64
        else:assert not EVENTS
    bad=[]
    for side in range(4):
        base=inputs(2,3,5)[side]
        vals=[None,Tensor(base.data,cuda=False),Tensor(base.data,device=1),Tensor(torch._neg_view(base.data)),Tensor(base.data.double())]
        if base.ndim>1:vals += [Tensor(base.data.transpose(-1,-2))]
        for value in vals:
            case=inputs(2,3,5);case[side]=value;bad.append(case)
    for ids in ([-2,0],[3,0],[2**31-1,-1]):bad.append(inputs(2,3,5,ids=ids))
    for scale in (1,True,float('nan'),float('inf'),float('-inf')):bad.append(inputs(2,3,5,scale=scale))
    case=inputs(2,3,5);case[3]=Tensor(torch.tensor([-1,-1,0],dtype=torch.int32)[::2]);bad.append(case)
    case=inputs(2,3,5);case[1]=Tensor(case[1].data.bfloat16());bad.append(case)
    case=inputs(2,3,5);case[1]=Tensor(torch.randn(3,2,3,5,dtype=torch.float16));bad.append(case)
    case=inputs(2,3,8);case[2]=Tensor(case[0].data.view(torch.float32).view(3,4));bad.append(case)
    case=inputs(2,3,8);case[2]=Tensor(case[1].data.view(torch.float32).view(6,6));bad.append(case)
    case=inputs(2,3,5);case[2].data.zero_();case[3]=Tensor(case[2].data.view(torch.int32).flatten()[:2]);bad.append(case)
    for number,case in enumerate(bad):
        EVENTS.clear();snap=[(t,raw(t.data).clone()) for t in case[:4] if isinstance(t,Tensor) and t.layout==torch.strided]
        try:fn(*case)
        except (ValueError,TypeError):pass
        else:raise AssertionError(f'invalid call {number} admitted')
        assert not EVENTS
        for t,saved in snap:assert torch.equal(raw(t.data),saved)
    case=inputs(2,3,5,ids=[0,1]);before=[raw(t.data).clone() for t in case[:4]];EVENTS.clear();launch.fail=True
    try:fn(*case)
    except RuntimeError:pass
    else:raise AssertionError('failure swallowed')
    assert active[0]==2 and EVENTS[-1]==('exit',2)
    for j in (0,1,3):assert torch.equal(raw(case[j].data),before[j])
    changed=raw(case[2].data)!=before[2];allowed=torch.zeros_like(changed);allowed[:12]=True
    assert not torch.any(changed&~allowed)
    print(f'{len(good)} complete direct cases, {len(bad)} rejection cases and partial atomic-write frame passed.')
    print('Both input formats, K=0 and tails, 64 splits and varied orders, sentinel/empty paths, direct identities and complete caller byte frames. CPU AST execution only.')

if __name__=='__main__':source_binding();host_checks()
