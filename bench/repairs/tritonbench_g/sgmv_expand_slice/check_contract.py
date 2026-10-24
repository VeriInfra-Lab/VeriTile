"""Actual checked host/kernel AST regressions; CPU arithmetic is not GPU conformance."""
import ast
import contextlib
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import triton

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'sgmv_expand_slice_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/sgmv_expand_slice/sgmv_expand_slice.py'
spec=importlib.util.spec_from_file_location('bgmv_helpers',HERE.parent/'bgmv_expand_slice/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,Pointer,typedraw=helpers.function,helpers.raw,helpers.Pointer,helpers.typedraw
EVENTS=[]

class Tensor(helpers.Tensor):
    def tolist(self):EVENTS.append(('tolist',self));return self.data.tolist()
    def element_size(self):return self.data.element_size()
    def squeeze(self,dim):
        t=Tensor(self.data.squeeze(dim),self.is_cuda,self.device.index or 0);EVENTS.append(('squeeze',self,t));return t

def kernel_ast(path):
    node=function(path,'_sgmv_expand_slice_kernel_repaired');node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    class Promotion(ast.NodeTransformer):
        # Triton augmented assignment rebinds a promoted SSA value. Torch's
        # in-place += would instead round it back to the old FP16 dtype.
        def visit_AugAssign(self,n):
            return ast.copy_location(ast.Assign(targets=[n.target],value=ast.BinOp(left=ast.Name(id=n.target.id,ctx=ast.Load()),op=n.op,right=n.value)),n)
    node=Promotion().visit(node);pid=[0]
    def offsets(p,mask):
        off=torch.as_tensor(p.offsets,dtype=torch.int64)
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.broadcast_to(torch.as_tensor(mask,dtype=torch.bool),off.shape)
        assert torch.all(off>=0) and torch.all(off<p.tensor.numel()),'formed pointer outside original view'
        ix=p.tensor.storage_offset()+off[mask]
        assert torch.all(ix>=0) and torch.all(ix<typedraw(p.tensor).numel()),'active access outside original backing'
        return off,mask,ix
    def load(p,mask=None,other=0):
        off,mask,ix=offsets(p,mask);out=torch.full_like(off,other,dtype=p.tensor.dtype);out[mask]=typedraw(p.tensor)[ix];return out
    def store(p,v,mask=None):
        off,mask,ix=offsets(p,mask);typedraw(p.tensor)[ix]=torch.broadcast_to(v,off.shape)[mask].to(p.tensor.dtype)
    tl=SimpleNamespace(program_id=lambda axis:torch.tensor(pid[0]),arange=torch.arange,int64=torch.int64,float32=torch.float32,
      cast=lambda v,dtype:torch.as_tensor(v,dtype=dtype),cdiv=lambda a,b:(a+b-1)//b,zeros=torch.zeros,where=torch.where,
      load=load,store=store,dot=lambda a,b,acc:a.float()@b.float()+acc)
    scope={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope['_sgmv_expand_slice_kernel_repaired'],pid

def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    for key,path in [('original_sha256',ORIGINAL),('candidate_sha256',SOURCE),('quotation_sha256',HERE/'Kernels.lean')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[key]
    assert hashlib.sha256(ast.dump(function(SOURCE,'_sgmv_expand_slice_kernel_repaired')).encode()).hexdigest()==pins['historical_kernel_ast_sha256']
    assert json.dumps(ast.dump(ast.parse(SOURCE.read_text()))) in (HERE/'Proofs/Direct/Compiler.lean').read_text()
    wrapper=function(SOURCE,'_sgmv_expand_slice_repaired')
    calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'clone','contiguous','resolve_neg','copy_','index_copy_','zeros','empty'}
    dot=[n for n in ast.walk(function(SOURCE,'_sgmv_expand_slice_kernel_repaired')) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dot']
    assert len(dot)==1 and len(dot[0].args)==3 and isinstance(dot[0].args[2],ast.Name) and dot[0].args[2].id=='accumulator'
    print('Historical dot-accumulator kernel AST unchanged; direct production AST and quotation pinned.')

def host_checks():
    torch.manual_seed(43)
    node=function(SOURCE,'_sgmv_expand_slice_repaired');node.decorator_list=[]
    scope={'torch':torch,'triton':triton};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
    k,pid=kernel_ast(SOURCE);active=[2]
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
                options={n:v for n,v in kw.items() if n not in ('num_warps','num_ctas')}
                for p in range(grid[0]):
                    pid[0]=p;k(*values,**options)
                    if self.fail:raise RuntimeError('injected partial launch')
            return run
    launch=Launch();scope['_sgmv_expand_slice_kernel_repaired']=launch
    scope['torch']=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float16=torch.float16,float32=torch.float32,
      int32=torch.int32,int64=torch.int64,cuda=SimpleNamespace(device=device))
    fn=scope['_sgmv_expand_slice_repaired']
    def inputs(n=5,k=17,four=False,metadata=None):
        starts,lengths,ids=metadata if metadata is not None else ([1,2,4,99],[3,3,0,7],[0,1,99,-1])
        b=len(starts);tokens=max([8]+[st+le for st,le,i in zip(starts,lengths,ids) if le>0 and i!=-1]);
        x=Tensor(torch.randn(tokens,k)*.2);w=Tensor(torch.randn((3,1,n,k) if four else (3,n,k),dtype=torch.float16)*.2)
        y=Tensor(torch.randn(tokens+2,n+8));meta=[Tensor(torch.tensor(v,dtype=torch.int32)) for v in (starts,lengths,ids)]
        return [x,w,y,*meta,b,1,tokens,4,n,True]
    good=[inputs(n,k,four) for n,k in ((1,0),(3,1),(5,15),(33,16),(35,17)) for four in (False,True)]
    good += [inputs(metadata=([],[],[])),inputs(n=0),inputs(metadata=([-99,0],[4,0],[-1,999])),inputs(n=33,k=17,metadata=([1,20],[35,33],[0,2]))]
    for case in good[:6]:good.append([*case[:11],False])
    for widths in itertools.product((torch.int32,torch.int64),repeat=3):
        case=inputs()
        for j,dtype in enumerate(widths,3):case[j]=Tensor(case[j].data.to(dtype))
        good.append(case)
    case=inputs();case[4]=Tensor(torch.tensor([3,3,0,7,-999],dtype=torch.int64));good.append(case)
    case=inputs();count=case[0].numel();case[0]=Tensor(torch.randn(count+8)[4:4+count].view(case[0].shape));good.append(case)
    case=inputs();count=case[2].numel();case[2]=Tensor(torch.randn(count+8)[4:4+count].view(case[2].shape));good.append(case)
    case=inputs(k=17);shared=torch.randn(max(case[0].numel(),case[1].numel()))*.1
    case[0]=Tensor(shared[:case[0].numel()].view(case[0].shape));case[1]=Tensor(shared.view(torch.float16)[:case[1].numel()].view(case[1].shape));good.append(case)
    case=inputs(metadata=([0,1],[0,1],[0,1]));case[5]=Tensor(case[3].data);good.append(case)
    def mask_for(args,only_first_tile=False):
        x,w,y,starts,lengths,ids,b,hint,tokens,offset,n,add=args
        allowed=torch.zeros(raw(y.data).numel(),dtype=torch.bool)
        for st,le,slot in zip(starts.data.tolist(),lengths.data.tolist()[:b],ids.data.tolist()):
            if le<=0 or slot==-1:continue
            for r in range(st,st+(min(le,32) if only_first_tile else le)):
                for c in range(offset,offset+(min(n,32) if only_first_tile else n)):
                    ix=y.data.storage_offset()+r*y.shape[1]+c;allowed[4*ix:4*(ix+1)]=True
            if only_first_tile:break
        return allowed
    for args in good:
        x,w,y,starts,lengths,ids,b,hint,tokens,offset,n,add=args
        before=[raw(t.data).clone() for t in args[:6]];expected=y.data.clone();weights=w.data.squeeze(1) if w.ndim==4 else w.data
        for st,le,slot in zip(starts.data.tolist(),lengths.data.tolist()[:b],ids.data.tolist()):
            if le<=0 or slot==-1:continue
            acc=torch.zeros(le,n)
            for block in range((x.shape[1]+15)//16):
                lo=block*16;hi=min(lo+16,x.shape[1]);acc += x.data[st:st+le,lo:hi].half().float()@weights[slot,:,lo:hi].float().T
            value=acc.half().float()
            if add:value=expected[st:st+le,offset:offset+n]+value
            expected[st:st+le,offset:offset+n]=value
        EVENTS.clear();result=fn(*args)
        assert result is None and active[0]==2
        torch.testing.assert_close(y.data,expected,rtol=2e-3,atol=2e-3,equal_nan=True)
        for j,t in enumerate(args[:6]):
            if j!=2:assert torch.equal(raw(t.data),before[j])
        allowed=mask_for(args);assert torch.equal(raw(y.data)[~allowed],before[2][~allowed])
        launches=[e for e in EVENTS if e[0]=='launch'];expected_batches=[j for j,(le,slot) in enumerate(zip(lengths.data.tolist()[:b],ids.data.tolist())) if le>0 and slot!=-1 and n]
        assert len(launches)==len(expected_batches)
        for e,batch in zip(launches,expected_batches):
            _,grid,actual,kw,dev=e
            assert grid==(triton.cdiv(int(lengths.data[batch]),32)*triton.cdiv(n,32),)
            assert actual[0] is x and actual[2] is y and all(actual[j+2] is args[j] for j in range(3,6))
            assert actual[1].untyped_storage()._cdata==w.untyped_storage()._cdata
            assert actual[8:]==(x.shape[1],1,n*x.shape[1],x.shape[1],1,y.shape[1],1,offset,batch)
        if not b:assert not EVENTS
        if not expected_batches:assert not any(e[0]=='enter' for e in EVENTS)
    bad=[]
    for side in range(6):
        base=inputs()[side]
        vals=[None,Tensor(base.data,cuda=False),Tensor(base.data,device=1),Tensor(torch._neg_view(base.data)),Tensor(base.data.double())]
        if base.ndim>1:vals += [Tensor(base.data.transpose(-1,-2))]
        for value in vals:
            case=inputs();case[side]=value;bad.append(case)
    for side,values in ((3,[-1,2,4,99]),(4,[-1,3,0,7]),(5,[-2,1,99,-1]),(5,[3,1,99,-1]),(4,[8,3,0,7])):
        case=inputs();case[side]=Tensor(torch.tensor(values,dtype=torch.int32));bad.append(case)
    for pos,value in ((6,3),(6,-1),(7,-1),(7,False),(8,7),(9,-1),(9,99),(10,4),(10,5.),(11,1)):
        case=inputs();case[pos]=value;bad.append(case)
    case=inputs();case[3]=Tensor(torch.tensor([1,99,2,99,4,99,99,99],dtype=torch.int32)[::2]);bad.append(case)
    case=inputs();case[1]=Tensor(torch.randn(3,2,5,17,dtype=torch.float16));bad.append(case)
    case=inputs();case[0]=Tensor(torch.randn(case[0].numel()+1)[1:].view(case[0].shape));bad.append(case)
    case=inputs(n=0);case[4]=Tensor(torch.tensor([-1,3,0,7],dtype=torch.int32));bad.append(case)
    # Output shares the full backing with each read role, even disjoint views.
    for role in (0,1,3,4,5):
        case=inputs();tensor=case[role].data;y=case[2].data
        shared=torch.empty(raw(tensor).numel()+raw(y).numel()+32,dtype=torch.uint8)
        read=shared[:raw(tensor).numel()].view(tensor.dtype).view(tensor.shape);read.copy_(tensor)
        off=(raw(tensor).numel()+15)//16*16
        output=shared[off:off+raw(y).numel()].view(y.dtype).view(y.shape);output.copy_(y)
        case[role]=Tensor(read);case[2]=Tensor(output);bad.append(case)
    for number,case in enumerate(bad):
        EVENTS.clear();saved=[(t,raw(t.data).clone()) for t in case[:6] if isinstance(t,Tensor) and t.layout==torch.strided]
        try:fn(*case)
        except (ValueError,TypeError):pass
        else:raise AssertionError(f'invalid call {number} admitted')
        assert not any(e[0] in ('enter','launch','squeeze') for e in EVENTS)
        for t,entry_bytes in saved:assert torch.equal(raw(t.data),entry_bytes)
    case=inputs(n=33,k=17,metadata=([1,20],[35,33],[0,2]));saved=[raw(t.data).clone() for t in case[:6]];launch.fail=True;EVENTS.clear()
    try:fn(*case)
    except RuntimeError:pass
    else:raise AssertionError('launch failure swallowed')
    assert active[0]==2 and EVENTS[-1]==('exit',2)
    for j in (0,1,3,4,5):assert torch.equal(raw(case[j].data),saved[j])
    changed=raw(case[2].data)!=saved[2];assert not torch.any(changed&~mask_for(case,True))
    print(f'{len(good)} complete direct cases, {len(bad)} rejection cases, serial overlaps and partial-write failure frame passed.')
    print('All metadata width combinations, both add modes, K0/tails, M/N tails, hint independence, aliases, aligned views and original caller bytes checked. CPU AST execution only.')

if __name__=='__main__':source_binding();host_checks()
