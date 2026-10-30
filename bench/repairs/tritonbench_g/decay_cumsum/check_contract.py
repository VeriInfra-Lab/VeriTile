"""Actual source AST/storage replay; CUDA launch and allocation are mocked."""
import ast
import contextlib
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'decay_cumsum_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/decay_cumsum/decay_cumsum.py'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
shared=load(HERE.parent/'add_example/check_contract.py','decay_storage_helpers')
function,raw_storage=shared.function,shared.raw_storage

class Tensor(shared.TensorFacade):
    def __getitem__(self,index):return Tensor(self.data[index],str(self.device))
    def copy_(self,other):self.data.copy_(other.data);return self

class Pointer:
    dtype=SimpleNamespace(element_ty=torch.float32)
    def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
    def __add__(self,n):return Pointer(self.tensor,self.offset+n)
    def __sub__(self,n):return Pointer(self.tensor,self.offset-n)

class Scalar(int):
    def to(self,dtype):assert dtype==torch.int64;return int(self)

class StripCasts(ast.NodeTransformer):
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and len(n.args)==1 and ast.unparse(n.args[0])=='tl.int64':return n.func.value
        if isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='tl.cast' and ast.unparse(n.args[1])=='tl.int64':return n.args[0]
        return n

def source_checks():
    for name in ('fwd_decay_cumsum','prepare_qg_kg','bwd_decay_global_cumsum'):
        new=function(SOURCE,name);old=function(ORIGINAL,name)
        # Each original tuple of IDs was spelled as three widened assignments.
        split=new.body[:3]
        assert all(isinstance(n,ast.Assign) for n in split)
        new.body=[ast.Assign(targets=[ast.Tuple(elts=[n.targets[0] for n in split],ctx=ast.Store())],
            value=ast.Tuple(elts=[n.value for n in split],ctx=ast.Load()))]+new.body[3:]
        new=StripCasts().visit(new)
        assert ast.dump(new)==ast.dump(old),name
    text=SOURCE.read_text()
    assert 'resolve_neg' not in text and '_debug_has_internal_overlap' not in text
    assert 'head_stride = T * width' in text and 'BT=remainder' in text
    assert 'destination.copy_(packed[..., :DK])' in text
    assert 'with torch.cuda.device(buffers[0].device):' in text
    lean=(HERE/'Contract.lean').read_text()
    for part in ['x.shape==[h.B,h.H,h.T,h.DK]','x.isContiguous','!x.isNeg','h.BK>1048576','h.features>=2^31','h.T/h.BT>65535','h.B*h.H>65535']:
        assert part in lean,part
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() in (HERE/'Proofs/NativeCompilation.lean').read_text()
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    assert 'Correct' not in (HERE/'FPEquiv.lean').read_text()
    assert 'decay_cumsum.Correct' not in (HERE/'Proofs/NativePublic.lean').read_text()

def interpreter(name,pid):
    def indices(p,mask=None):
        off=torch.as_tensor(p.offset,dtype=torch.int64)
        if mask is None:mask=torch.ones_like(off,dtype=torch.bool)
        ix=p.tensor.storage_offset()+off[mask]
        assert bool((ix>=0).all()) and bool((ix<raw_storage(p.tensor).numel()).all()),'native storage bounds'
        return off,mask,ix
    def load_word(p,mask=None,other=0):
        off,mask,ix=indices(p,mask);out=torch.full_like(off,other,dtype=torch.float32);out[mask]=raw_storage(p.tensor)[ix];return out
    def store_word(p,value,mask):
        _,mask,ix=indices(p,mask);raw_storage(p.tensor)[ix]=value[mask]
    node=function(SOURCE,name);node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    ns={'inv_ln2':torch.tensor(1.44269504,dtype=torch.float32),'tl':SimpleNamespace(program_id=lambda ax:Scalar(pid[ax]),
        int64=torch.int64,float32=torch.float32,arange=torch.arange,zeros=torch.zeros,cast=lambda x,dtype:int(x),
        load=load_word,store=store_word,math=SimpleNamespace(exp2=torch.exp2))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[name]

def reference(kind,values,bt,scale):
    names=['g','go'] if kind=='forward' else ['q','k','g','qg','kg'] if kind=='prepare' else ['dqi','dqo','dki','dko','q','k','g','dg']
    v=dict(zip(names,[x.clone() for x in values]));t=values[0].shape[2]
    for start in range(0,t,bt):
        stop=min(start+bt,t)
        if kind=='forward':
            acc=torch.zeros_like(v['g'][:,:,0,:])
            for time in range(start,stop):acc=acc+v['g'][:,:,time,:]*torch.tensor(1.44269504);v['go'][:,:,time,:]=acc
        elif kind=='prepare':
            last=v['g'][:,:,stop-1,:]
            for time in range(start,stop):
                g=v['g'][:,:,time,:];v['qg'][:,:,time,:]=v['q'][:,:,time,:]*(torch.exp2(g)*scale)
                v['kg'][:,:,time,:]=v['k'][:,:,time,:]*torch.exp2(last-g)
        else:
            last=v['g'][:,:,stop-1,:];acc=torch.zeros_like(last)
            for time in reversed(range(start,stop)):
                g=v['g'][:,:,time,:]
                dq=v['dqi'][:,:,time,:]+v['dqo'][:,:,time,:]*torch.exp2(g)
                dk=v['dki'][:,:,time,:]+v['dko'][:,:,time,:]*torch.exp2(last-g)
                v['dqo'][:,:,time,:]=dq;v['dko'][:,:,time,:]=dk
                acc=acc+(dq*v['q'][:,:,time,:]-dk*v['k'][:,:,time,:]);v['dg'][:,:,time,:]=acc
    return v

def checks():
    module=load(SOURCE,'decay_checked_review');pid=[0,0,0];calls=[];allocations=[];executions=[0]
    def allocate(shape,dtype,device,zero=False):
        data=torch.zeros(shape,dtype=dtype) if zero else torch.empty(shape,dtype=dtype)
        allocations.append(data);return Tensor(data,str(device))
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        zeros=lambda shape,dtype,device:allocate(shape,dtype,device,True),empty=allocate,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    counts={'fwd_decay_cumsum':2,'prepare_qg_kg':5,'bwd_decay_global_cumsum':8}
    for name,nargs in counts.items():
        kernel=interpreter(name,pid)
        class Launch:
            def __init__(self,k,n):self.k,self.n=k,n
            def __getitem__(self,grid):
                def run(*args,**kw):
                    calls.append((grid,kw.copy()))
                    ptrs=[Pointer(x.data) for x in args[:self.n]]
                    for row in reversed(range(grid[2])):
                        for chunk in reversed(range(grid[1])):
                            for feature in reversed(range(grid[0])):
                                pid[:]=[feature,chunk,row];self.k(*ptrs,*args[self.n:],**kw);executions[0]+=1
                return run
        setattr(module,name,Launch(kernel,nargs))
    good=0
    for shape,bt,bk in [((2,2,5,5),2,4),((1,1,1,1),8,1),((1,2,4,8),2,4),((2,1,3,7),2,8),
        ((0,2,5,5),2,4),((2,0,5,5),2,4),((2,2,0,5),2,4),((2,2,5,0),2,4)]:
        for kind,name,n in [('forward','launch_fwd_decay_cumsum',2),('prepare','launch_prepare_qg_kg',5),('backward','launch_bwd_decay_global_cumsum',8)]:
            numel=1
            for d in shape:numel*=d
            backing=[torch.arange(numel+16,dtype=torch.float32)/64+i/16 for i in range(n)]
            values=[x[8:8+numel].reshape(shape) for x in backing]
            tensors=[Tensor(x) for x in values];before=[x.clone() for x in values];guards=[x.clone() for x in backing]
            expected=reference(kind,values,bt,0.5);calls.clear();allocations.clear()
            result=getattr(module,name)(*tensors,*shape[:3],0.5,bt,bk,shape[3]);assert result is None
            output_names=['go'] if kind=='forward' else ['qg','kg'] if kind=='prepare' else ['dqo','dko','dg']
            names=['g','go'] if kind=='forward' else ['q','k','g','qg','kg'] if kind=='prepare' else ['dqi','dqo','dki','dko','q','k','g','dg']
            for i,role in enumerate(names):
                torch.testing.assert_close(values[i],expected[role] if role in output_names else before[i],rtol=1e-6,atol=1e-6)
                torch.testing.assert_close(backing[i][:8],guards[i][:8]);torch.testing.assert_close(backing[i][-8:],guards[i][-8:])
            assert bool(calls)==bool(numel) and bool(allocations)==bool(numel)
            if numel:assert all(tuple(x.shape)==(*shape[:3],((shape[3]+bk-1)//bk)*bk) for x in allocations)
            good+=1
    ordinary=Tensor(torch.ones(1,1,2,3))
    cases=[([ordinary],(1,1,2,3,1.0,2,4),TypeError)]
    bad=[([],ordinary,1,1,2,1.,2,4,3),(Tensor(torch.ones(1,1,2,3,dtype=torch.float64)),ordinary,1,1,2,1.,2,4,3),
        (Tensor(torch._neg_view(torch.ones(1,1,2,3))),ordinary,1,1,2,1.,2,4,3),
        (Tensor(torch.ones(1,1,3,2).transpose(2,3)),ordinary,1,1,2,1.,2,4,3),
        (ordinary,ordinary,1,1,2,1.,0,4,3),(ordinary,ordinary,1,1,2,1.,2,3,3),
        (ordinary,ordinary,1,1,2,1,2,4,3),(ordinary,ordinary,1,1,3,1.,2,4,3),
        (Tensor(torch.ones(1,1,2,3),'cpu',False),ordinary,1,1,2,1.,2,4,3),
        (Tensor(torch.ones(1,1,2,3),'cuda:1'),ordinary,1,1,2,1.,2,4,3)]
    bad.append((Tensor(shared.unaligned_tensor(1).reshape(1,1,1,4)),Tensor(torch.ones(1,1,1,4)),1,1,1,1.,1,4,4))
    class MetaTensor(Tensor):
        def data_ptr(self):return 4
    for shape,bt,bk in [((1,65536,1,1),1,1),((1,1,65536,1),1,1),((1,1,1,2**31),1,1)]:
        huge=MetaTensor(torch.empty(shape,device='meta'))
        bad.append((huge,huge,*shape[:3],1.,bt,bk,shape[3]))
    for args in bad:
        calls.clear();allocations.clear()
        try:module.launch_fwd_decay_cumsum(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported input admitted')
        assert not calls and not allocations
    # Aliased destinations retain entry snapshots and source writeback order.
    source=Tensor(torch.arange(15,dtype=torch.float32).reshape(1,1,3,5)/64)
    expected=reference('forward',[source.data,source.data],2,1.)['go']
    module.launch_fwd_decay_cumsum(source,source,1,1,3,1.,2,4,5)
    torch.testing.assert_close(source.data,expected,rtol=1e-6,atol=1e-6)
    q=Tensor(torch.ones(1,1,3,5));k=Tensor(torch.ones(1,1,3,5)*2);g=Tensor(torch.zeros(1,1,3,5));same=Tensor(torch.zeros(1,1,3,5))
    expected=reference('prepare',[q.data,k.data,g.data,same.data,same.data],2,0.5)['qg']
    module.launch_prepare_qg_kg(q,k,g,same,same,1,1,3,0.5,2,4,5)
    torch.testing.assert_close(same.data,expected,rtol=0,atol=0)
    good+=2
    # Original H*T*DK head stride overshoots the final flattened row.
    b,h,t,dk=2,2,4,8
    assert (b*h-1)*(h*t*dk)+(t-1)*dk+dk>b*h*t*dk
    assert 5//4*4<5 and 5//2*2<5
    return len(bad),good,executions[0]

if __name__=='__main__':
    source_checks();rejected,admitted,programs=checks()
    print(f'Passed source binding, {rejected} rejections, {admitted} phase/shape controls, {programs} program executions.')
    print('CPU AST/storage replay with mocked CUDA; native FMA/exp2 conformance is not claimed.')
