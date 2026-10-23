"""Actual AST/CPU storage controls for the guarded chunked-scan wrapper."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'chunked_cumsum_fwd_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunked_cumsum_fwd/chunked_cumsum_fwd.py'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
shared=load(HERE.parent/'add_example/check_contract.py','chunked_storage_helpers')
function,raw_storage=shared.function,shared.raw_storage
class Tensor(shared.TensorFacade):
    @property
    def ndim(self):return self.data.ndim
    def stride(self,i=None):return self.data.stride() if i is None else self.data.stride(i)
class Scalar(int):
    def to(self,dtype):return int(self)
class Pointer:
    def __init__(self,tensor,offsets=0):self.tensor,self.offsets=tensor,offsets
    def __add__(self,other):return Pointer(self.tensor,self.offsets+other)
class StripCasts(ast.NodeTransformer):
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and len(n.args)==1 and ast.unparse(n.args[0])=='tl.uint64':return n.func.value
        if isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='tl.cast' and ast.unparse(n.args[1])=='tl.uint64':return n.args[0]
        return n

def source_checks():
    new=StripCasts().visit(function(SOURCE,'_chunk_cumsum_fwd_kernel'))
    new.body=[n for n in new.body if not(isinstance(n,ast.Assign) and len(n.targets)==1 and ast.dump(n.targets[0]).replace('Store()','Load()')==ast.dump(n.value))]
    assert ast.dump(new)==ast.dump(function(ORIGINAL,'_chunk_cumsum_fwd_kernel'))
    source=SOURCE.read_text()
    assert 'resolve_neg' not in source and '.contiguous()' not in source
    assert 'nchunks = -(-seqlen // chunk_size)' in source
    assert 'return dA_cumsum, dt_out' in source
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    assert 'FPEquiv' not in (HERE/'Correct.lean').read_text()
    assert 'Correct' not in (HERE/'FPEquiv.lean').read_text()
    host=(HERE/'Proofs/Host.lean').read_text()
    assert 'memoryOriginal : memory=a.memory' in host and 'dtOriginal : dt=call.dt' in host
    assert 'resolutionFailure' not in host
    lean=(HERE/'Contract.lean').read_text()
    for part in ['x.dtype==.floating .fp32','!x.isNeg','chunk<=0','chunk>=2^63','dt.shape.getD 0 0>=2^31','dt.shape.getD 2 0>65535']:
        assert part in lean

def interpreter():
    pid=[0,0,0]
    def load_word(p,mask,other=0):
        offsets,mask=torch.broadcast_tensors(torch.as_tensor(p.offsets),mask)
        ix=p.tensor.storage_offset()+offsets[mask]
        assert bool((ix>=0).all()) and bool((ix<raw_storage(p.tensor).numel()).all())
        out=torch.full_like(offsets,other,dtype=torch.float32);out[mask]=raw_storage(p.tensor)[ix];return out
    def store_word(p,values,mask):
        offsets,values,mask=torch.broadcast_tensors(p.offsets,values,mask)
        ix=p.tensor.storage_offset()+offsets[mask]
        assert bool((ix>=0).all()) and bool((ix<raw_storage(p.tensor).numel()).all())
        raw_storage(p.tensor)[ix]=values[mask]
    node=function(SOURCE,'_chunk_cumsum_fwd_kernel');node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    ns={'tl':SimpleNamespace(program_id=lambda axis:Scalar(pid[axis]),uint64=torch.int64,float32=torch.float32,
        arange=torch.arange,cast=lambda x,dtype:int(x),load=load_word,store=store_word,exp=torch.exp,log=torch.log,
        where=torch.where,minimum=lambda x,y:torch.minimum(x,torch.tensor(y,dtype=torch.float32)),
        maximum=lambda x,y:torch.maximum(x,torch.tensor(y,dtype=torch.float32)),cumsum=lambda x,axis:torch.cumsum(x,axis))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['_chunk_cumsum_fwd_kernel'],pid

def reference(dt,rate,chunk,bias,softplus,limits):
    batch,sequence,heads=dt.shape;nchunks=(sequence+chunk-1)//chunk
    out=torch.zeros(batch,heads,nchunks,chunk)
    for k in range(nchunks):
        size=min(chunk,sequence-k*chunk);values=dt[:,k*chunk:k*chunk+size,:].transpose(1,2).clone()
        if bias is not None:values=values+bias[None,:,None]
        if softplus:values=torch.where(values<=20,torch.log(1+torch.exp(values)),values)
        values=torch.minimum(torch.maximum(values,torch.tensor(limits[0])),torch.tensor(limits[1]))
        out[:,:,k,:size]=values
    return (out*rate[None,:,None,None]).cumsum(3),out

def checks():
    module=load(SOURCE,'chunked_contract_review');kernel,pid=interpreter();calls=[];backings=[];count=[0]
    def empty(*shape,device,dtype):
        n=1
        for d in shape:n*=d
        storage=torch.full((n+16,),-12345.,dtype=dtype);backings.append(storage)
        calls.append('allocation');return Tensor(storage[8:8+n].reshape(shape),str(device))
    class Launch:
        def __getitem__(self,grid):
            def run(dt,rate,bias,out,total,*args,**kw):
                calls.append((dt,rate,bias,out,total))
                for bh in (1,2,4,8,16,32,64):
                    config=dict(kw,BLOCK_SIZE_H=bh);dims=grid(config)
                    assert dims==(dt.shape[0],(dt.shape[1]+args[3]-1)//args[3],(dt.shape[2]+bh-1)//bh)
                    for b in reversed(range(dims[0])):
                        for k in reversed(range(dims[1])):
                            for h in reversed(range(dims[2])):
                                pid[:]=[b,k,h]
                                kernel(Pointer(dt.data),Pointer(rate.data),None if bias is None else Pointer(bias.data),
                                    Pointer(out.data),Pointer(total.data),*args,**config);count[0]+=1
            return run
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,empty=empty,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    module._chunk_cumsum_fwd_kernel=Launch()
    cases=[]
    for shape,chunk in [((2,10,4),5),((1,7,3),3),((2,1,1),1),((1,17,9),8),((0,7,3),4),((2,0,3),4),((1,4,0),3)]:
        n=shape[0]*shape[1]*shape[2];dt=(torch.arange(n,dtype=torch.float32).reshape(shape)%13-6)/8
        cases.append((dt,torch.arange(shape[2],dtype=torch.float32)/8,chunk))
    backing=torch.arange(2*7*6,dtype=torch.float32).reshape(2,7,6)/64
    cases += [(backing[:,:,::2],torch.ones(6)[::2],4),(torch.ones(1,7,1).expand(2,7,3),torch.ones(1).expand(3),4)]
    admitted=0
    for dt,rate,chunk in cases:
        for has_bias in (False,True):
            for softplus in (False,True):
                bias=torch.arange(rate.numel(),dtype=torch.float32)/16 if has_bias else None
                calls.clear();backings.clear();dtbefore=dt.clone();rbefore=rate.clone()
                x=Tensor(dt);a=Tensor(rate);b=None if bias is None else Tensor(bias)
                total,out=module._chunk_cumsum_fwd(x,a,chunk,dt_bias=b,dt_softplus=softplus)
                expected=reference(dt,rate,chunk,bias,softplus,(0.,float('inf')))
                torch.testing.assert_close(total.data,expected[0],rtol=1e-6,atol=1e-6);torch.testing.assert_close(out.data,expected[1],rtol=0,atol=0)
                torch.testing.assert_close(dt,dtbefore);torch.testing.assert_close(rate,rbefore)
                assert calls[:2]==['allocation','allocation'] and len(calls)==3
                assert calls[2][0] is x and calls[2][1] is a and calls[2][2] is b
                assert calls[2][3] is out and calls[2][4] is total
                assert all(bool((s[:8]==-12345).all()) and bool((s[-8:]==-12345).all()) for s in backings)
                admitted+=1
    x=Tensor(torch.ones(1,2,3));a=Tensor(torch.ones(3))
    bad=[([],a,2,{}),(x,[],2,{}),(x,a,2,{'dt_bias':[]}),
        (Tensor(torch.ones(1,2,3,dtype=torch.float64)),a,2,{}),
        (Tensor(torch._neg_view(torch.ones(1,2,3))),a,2,{}),
        (Tensor(torch.ones(1,2,3),'cpu',False),a,2,{}),
        (x,Tensor(torch.ones(3),'cuda:1'),2,{}),(Tensor(torch.ones(2,3)),a,2,{}),
        (x,Tensor(torch.ones(2)),2,{}),(x,a,0,{}),(x,a,-1,{}),(x,a,2**63,{}),(x,a,1048577,{}),
        (x,a,2,{'dt_softplus':1}),(x,a,2,{'dt_limit':[0.,1.]}),(x,a,2,{'dt_limit':(0,1.)})]
    bad.append((Tensor(shared.unaligned_tensor(1).reshape(1,2,2)),Tensor(torch.ones(2)),2,{}))
    class MetaTensor(Tensor):
        def data_ptr(self):return 4
    for shape,chunk in [((2**31,1,1),1),((1,65536,1),1),((1,1,65536),1)]:
        bad.append((MetaTensor(torch.empty(shape,device='meta')),MetaTensor(torch.empty(shape[2],device='meta')),chunk,{}))
    for dt,rate,chunk,kwargs in bad:
        calls.clear()
        try:module._chunk_cumsum_fwd(dt,rate,chunk,**kwargs)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported input admitted')
        assert calls==[]
    import math
    n=2**53+1
    assert math.ceil(n/1)!=-(-n//1)
    negative=torch._neg_view(torch.tensor([1.,2.]))
    assert not torch.equal(raw_storage(negative),negative)
    return len(bad),admitted,count[0]

if __name__=='__main__':
    source_checks();rejected,admitted,programs=checks()
    print(f'Passed unchanged floating AST, {rejected} rejections, {admitted} admissions, {programs} program executions over all seven configs.')
    print('CPU storage replay with mocked CUDA; native scan/FMA/scalar conformance remains deferred.')
