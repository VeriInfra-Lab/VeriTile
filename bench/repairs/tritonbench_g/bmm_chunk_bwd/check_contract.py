"""Execute the actual accepted arithmetic AST on CPU backing storage."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'bmm_chunk_bwd_repaired.py'
spec=importlib.util.spec_from_file_location('bwd_forward_helpers',HERE.parent/'bmm_chunk_fwd/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
Tensor,Pointer,raw,backing_bytes=f.Tensor,f.Pointer,f.raw,f.backing_bytes
CONFIGS=f.CONFIGS

def source_binding():
    digest=hashlib.sha256(ast.dump(f.function(SOURCE,'_bmm_chunk_bwd_kernel'),include_attributes=False).encode()).hexdigest()
    assert digest=='67520e1af930adfb65824574210b578bc3f1e4e123ab013f5a6c9dc64b09d794'
    wrapper=f.function(SOURCE,'_bmm_chunk_bwd')
    assert len([n for n in wrapper.body if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body)])==17
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'clone','resolve_neg','copy_'} for n in ast.walk(wrapper))
    lean=(HERE/'Contract.lean').read_text().split('def checkInputs',1)[1].split('def checkArguments',1)[0]
    assert re.findall(r'\.error \.(\w+)',lean)==['layout','dtype','cuda','device','ranks','chunk','shapes','residual','output','contiguous','strides','negative','extent','grid','alignment','overlap']

def main():
    source_binding()
    spec=importlib.util.spec_from_file_location('bwd_direct',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    kernel,pids=f.interpreter(SOURCE,'_bmm_chunk_bwd_kernel')
    calls=[];allocations=[];choice=[0]
    class Launch:
        def __getitem__(self,grid):
            def run(a,d,out,res,*args,**kw):
                for ci in (0,6,choice[0]):
                    m,n,cs,_,_=CONFIGS[ci];dims=grid({'BLOCK_SIZE_M':m,'BLOCK_SIZE_N':n})
                    calls.append((ci,dims,a,d,out,res))
                    for y in range(dims[1]):
                        for z in reversed(range(dims[2])):
                            for x in reversed(range(dims[0])):
                                pids[:]=[x,y,z]
                                kernel(Pointer(a.data),Pointer(d.data),Pointer(out.data),None if res is None else Pointer(res.data),*args,kw['HAS_RESIDUAL'],m,n,cs)
            return run
    def empty_like(a):
        storage=torch.full((a.numel()+9,),-777.,dtype=a.data.dtype)
        out=Tensor(storage[4:4+a.numel()].reshape(a.shape));allocations.append(out);return out
    module._bmm_chunk_bwd_kernel=Launch();module.tl=SimpleNamespace(float32=torch.float32)
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,empty_like=empty_like,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    run=module._bmm_chunk_bwd
    cases=[((1,5,3),3,False,False),((2,5,2,5),3,True,True),((1,7,1),4,True,False),
           ((1,3,0),2,False,True),((0,4,3),3,False,False),((2,0,3),2,True,False),
           ((1,4,0,3),3,False,True),((1,65,33),65,False,True),((1,1,1),1,True,True)]
    for ci,(shape,chunk,has_res,supplied) in enumerate(cases):
        count=1
        for x in shape:count*=x
        a=Tensor((torch.arange(count+5,dtype=torch.float32)[2:2+count]%5-2).reshape(shape))
        ch=(shape[1]+chunk-1)//chunk;groups=shape[2] if len(shape)==4 else 1
        dshape=(shape[0],ch,groups,chunk,chunk) if len(shape)==4 else (shape[0],ch,chunk,chunk)
        dc=shape[0]*ch*groups*chunk*chunk
        d=Tensor((torch.arange(dc+7,dtype=torch.float32)[3:3+dc]%7-3).reshape(dshape))
        res=Tensor(torch.full(shape,1.25)) if has_res else None
        out=empty_like(a) if supplied else None
        inputs=[a,d]+([] if res is None else [res]);before=[backing_bytes(t.data).clone() for t in inputs]
        choice[0]=ci%9;nc=len(calls);na=len(allocations)
        result=run(a,d,res,out)
        assert result is (out if supplied else allocations[-1])
        assert len(allocations)==na+(not supplied)
        expected=torch.zeros(shape)
        for b in range(shape[0]):
            for s in range(shape[1]):
                c,r=divmod(s,chunk);limit=min(chunk,shape[1]-c*chunk)
                for g in range(groups):
                    for n in range(shape[-1]):
                        value=0.
                        for k in range(limit):
                            dix=(b,c,g,k,r) if len(shape)==4 else (b,c,k,r)
                            aix=(b,c*chunk+k,g,n) if len(shape)==4 else (b,c*chunk+k,n)
                            value+=float(d.data[dix]*a.data[aix])
                        ix=(b,s,g,n) if len(shape)==4 else (b,s,n)
                        expected[ix]=value+(1.25 if has_res else 0.)
        assert torch.equal(result.data,expected),(ci,result.data,expected)
        assert all(torch.equal(x,backing_bytes(t.data)) for x,t in zip(before,inputs))
        backing=backing_bytes(result.data).view(torch.float32)
        assert torch.equal(backing[:4],torch.full((4,),-777.)) and torch.equal(backing[-5:],torch.full((5,),-777.))
        assert (len(calls)>nc)==bool(count)
        for entry in calls[nc:]:assert entry[2] is a and entry[3] is d and entry[4] is result and entry[5] is res
    rejected=0
    def fresh():return [Tensor(torch.ones(2,5,3)),Tensor(torch.ones(2,2,3,3)),Tensor(torch.ones(2,5,3)),Tensor(torch.ones(2,5,3))]
    def reject(args):
        nonlocal rejected
        nc,na=len(calls),len(allocations);before=[backing_bytes(x.data).clone() for x in args if isinstance(x,Tensor)]
        try:run(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid input admitted')
        assert (nc,na)==(len(calls),len(allocations))
        assert all(torch.equal(x,backing_bytes(t.data)) for x,t in zip(before,[x for x in args if isinstance(x,Tensor)]))
        rejected+=1
    for i in range(4):
        for mutation in ('object','layout','dtype','cpu','device','contiguous','strides','negative','extent','alignment'):
            args=fresh();x=args[i]
            if mutation=='object':args[i]=object()
            elif mutation=='layout':
                class Sparse(Tensor):
                    @property
                    def layout(self):return torch.sparse_coo
                args[i]=Sparse(x.data)
            elif mutation=='dtype':args[i]=Tensor(x.data.to(torch.float64))
            elif mutation=='cpu':x.is_cuda=False
            elif mutation=='device':x.device=torch.device('cuda:1')
            elif mutation=='contiguous':x.is_contiguous=lambda:False
            elif mutation=='strides':x.stride=lambda _:17
            elif mutation=='negative':x.is_neg=lambda:True
            elif mutation=='extent':x.numel=lambda:1<<63
            else:x.data_ptr=lambda:3
            reject(args)
    for i,shape in ((0,(2,5)),(1,(2,3,3)),(1,(2,2,3,0)),(1,(2,2,2,3)),(1,(2,3,3,3)),(2,(2,5,2)),(3,(2,5,2))):
        args=fresh();args[i]=Tensor(torch.ones(shape));reject(args)
    for i in range(4):
        for j in range(i+1,4):
            args=fresh();ptr=args[i].data_ptr();args[j].data_ptr=lambda ptr=ptr:ptr;reject(args)
    # Actual contiguous singleton strides must be pinned too.
    args=[Tensor(torch.ones(1,5,1).as_strided((1,5,1),(5,1,17))),Tensor(torch.ones(1,2,3,3)),None,None];reject(args)
    print(f'{len(cases)} admissions, {rejected} rejections; full contractions, residuals, tuner traces, supplied/fresh identities and backing-byte frames passed.')
    print('Accepted arithmetic AST retained; CPU arithmetic and CUDA launch mocked; no GPU numerical claim.')

if __name__=='__main__':main()
