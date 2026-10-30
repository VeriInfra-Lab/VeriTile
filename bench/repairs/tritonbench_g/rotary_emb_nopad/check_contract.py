"""Direct original-shape rotary kernels on CPU storage; CUDA is mocked."""
import ast
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'rotary_emb_nopad_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rotary_emb_nopad/rotary_emb_nopad.py'
spec=importlib.util.spec_from_file_location('transform_fixtures',HERE.parent/'rotary_transform/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class Tensor(f.Tensor):
    def size(self,axis=None):return self.data.size() if axis is None else self.data.size(axis)


def host(path=SOURCE,current_device=1):
    events,ids=[],[0,0]
    def load(ptr,mask=None,other=0):
        off=torch.as_tensor(ptr.offset)
        if mask is None:return f.storage(ptr.tensor)[ptr.tensor.storage_offset()+off]
        off,mask=torch.broadcast_tensors(off,torch.as_tensor(mask))
        result=torch.full(off.shape,other,dtype=ptr.tensor.dtype)
        result[mask]=f.storage(ptr.tensor)[ptr.tensor.storage_offset()+off[mask]]
        return result
    def store(ptr,value,mask=None):
        off=torch.as_tensor(ptr.offset)
        if mask is None:mask=torch.ones(off.shape,dtype=torch.bool)
        off,mask,value=torch.broadcast_tensors(off,torch.as_tensor(mask),value)
        f.storage(ptr.tensor)[ptr.tensor.storage_offset()+off[mask]]=value[mask]
    ns={'tl':SimpleNamespace(program_id=lambda axis:torch.tensor(ids[axis]),arange=torch.arange,
        int64=torch.int64,load=load,store=store)}
    launches={}
    for name in ('rotary_embedding_kernel','fused_rotary_embedding_kernel_v2'):
        fn=f.f.function(path,name);fn.decorator_list=[]
        for a in fn.args.args:a.annotation=None
        args=[a.arg for a in fn.args.args]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns)
        kernel=ns[name]
        def launcher(kernel=kernel,argnames=args,name=name):
            class Launch:
                def __getitem__(self,grid):
                    def run(*args,**kwargs):
                        params=dict(zip(argnames,args));params.update(kwargs)
                        dims=grid(params) if callable(grid) else grid
                        events.append((name,dims,kwargs['num_warps']))
                        kwargs.pop('num_warps')
                        args=[f.f.Pointer(t.data) if isinstance(t,Tensor) else t for t in args]
                        for head,token in itertools.product(*(reversed(range(n)) for n in dims)):
                            ids[:]=[head,token];kernel(*args,**kwargs)
                    return run
            return Launch()
        launches[name]=launcher()
    ns.update(launches)
    ns.update(torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int64=torch.int64,
        cuda=SimpleNamespace(current_device=lambda:current_device)),
        triton=SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()))
    body=[]
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_span','_check_inputs','rotary_embedding'):
            n.returns=None
            for a in n.args.args:a.annotation=None
            body.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),ns)
    return ns['rotary_embedding'],events


def reference(x,c,s):
    half=x.shape[-1]//2
    a,b=x[...,:half],x[...,half:]
    return torch.cat((a*c[:x.shape[0],None,:half]-b*s[:x.shape[0],None,:half],
        a*s[:x.shape[0],None,:half]+b*c[:x.shape[0],None,:half]),-1)


def main():
    rotate,events=host()
    for name in ('rotary_embedding_kernel','fused_rotary_embedding_kernel_v2'):
        def arithmetic(path):
            return {n.targets[0].id:ast.dump(n.value) for n in ast.walk(f.f.function(path,name))
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('out_')}
        assert arithmetic(SOURCE)==arithmetic(ORIGINAL)
    assert 'enable_fp_fusion' not in SOURCE.read_text()
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','copy_','resolve_neg')
        for n in ast.walk(f.f.function(SOURCE,'rotary_embedding')))
    torch.manual_seed(271)
    paths=0
    cases=[(1,1,1,2),(5,6,2,6),(7,4,1,10),(3,2,1,34),(1,2,1,256),(1,2,1,512),(0,4,2,6),(3,4,2,0)]
    for tokens,hq,hk,width in cases:
        for cached in (False,True):
            tensors=[];backing=[]
            for shape in ((tokens,hq,width),(tokens,hk,width),(tokens+1,width//2+2),(tokens+2,width//2+3)):
                n=math.prod(shape);raw=torch.randint(-3,4,(n+9,)).float()
                tensors.append(Tensor(raw[:n].reshape(shape)));backing.append(raw)
            q,k,c,s=tensors;q0=q.data.clone();k0=k.data.clone();raw0=[x.clone() for x in backing]
            qexpected,kexpected=reference(q0,c.data,s.data),reference(k0,c.data,s.data)
            kwargs={};cache=None
            if cached:
                block_size=4;blocks=(tokens+3)//4+1
                raw=torch.full((blocks*hk*block_size*width+9,),-97.)
                cache=Tensor(raw[:-9].reshape(blocks,hk,block_size,width))
                tables=torch.zeros((tokens,2),dtype=torch.int64)
                lengths=torch.tensor([t%4+1 for t in range(tokens)],dtype=torch.int64)
                for t in range(tokens):tables[t,0]=t//4
                kwargs=dict(k_cache=cache,block_tables=Tensor(tables),kv_lengths=Tensor(lengths))
                cache_expected=cache.data.clone()
                for t in range(tokens):cache_expected[t//4,:,t%4,:]=kexpected[t]
            events.clear();assert rotate(q,k,c,s,**kwargs) is None
            assert torch.equal(q.data,qexpected)
            assert torch.equal(k.data,k0 if cached else kexpected)
            if cached:assert torch.equal(cache.data,cache_expected) and torch.equal(raw[-9:],torch.full((9,),-97.))
            for i in range(4):
                assert torch.equal(backing[i][tensors[i].numel():],raw0[i][tensors[i].numel():])
            assert torch.equal(c.data,raw0[2][:-9].reshape(c.shape)) and torch.equal(s.data,raw0[3][:-9].reshape(s.shape))
            assert len(events)==bool(tokens*width)
            if events:assert events[0][2]==(16 if width>=512 else 8 if width>=256 else 4)
            paths+=1
    base=[Tensor(torch.randn(2,4,6)),Tensor(torch.randn(2,2,6)),Tensor(torch.randn(2,3)),Tensor(torch.randn(2,4))]
    failures=[]
    def bad(i,t):
        args=base.copy();args[i]=t;failures.append((args,{}))
    bad(0,[]);bad(0,Tensor(torch.randn(2,4,6,dtype=torch.float64)));bad(0,Tensor(torch._neg_view(base[0].data)))
    bad(0,Tensor(torch.randn(2,3,6)));bad(0,Tensor(torch.randn(2,4,5)));bad(0,Tensor(torch.randn(2,4,12)[...,::2]))
    bad(0,Tensor(base[0].data,'cpu',False));bad(3,base[2]);bad(1,Tensor(torch.randn(1,2,6)))
    cache=Tensor(torch.zeros(2,2,4,6));tables=Tensor(torch.tensor([[0],[1]],dtype=torch.int64));lengths=Tensor(torch.ones(2,dtype=torch.int64))
    valid=dict(k_cache=cache,block_tables=tables,kv_lengths=lengths)
    for field,value in [('kv_lengths',Tensor(torch.tensor([0,1]))),('kv_lengths',Tensor(torch.tensor([5,1]))),
        ('kv_lengths',Tensor(torch.ones(2,dtype=torch.int32))),('block_tables',Tensor(torch.tensor([[-1],[1]]))),
        ('block_tables',Tensor(torch.tensor([[2],[1]]))),('block_tables',Tensor(torch.tensor([[0],[0]]))),
        ('block_tables',Tensor(torch.zeros(1,1,dtype=torch.int64)))]:
        kwargs=valid.copy();kwargs[field]=value;failures.append((base,kwargs))
    for remainder in (1,2,3):
        storage=torch.zeros(200,dtype=torch.uint8).untyped_storage()[remainder:remainder+192]
        bad(0,Tensor(torch.empty(0).set_(storage,0,(2,4,6),(24,6,1))))
    for remainder in (1,2,3,4,5,6,7):
        storage=torch.zeros(32,dtype=torch.uint8).untyped_storage()[remainder:remainder+16]
        value=Tensor(torch.empty(0,dtype=torch.int64).set_(storage,0,(2,),(1,)))
        kwargs=valid.copy();kwargs['kv_lengths']=value;failures.append((base,kwargs))
    for args,kwargs in failures:
        events.clear()
        try:rotate(*args,**kwargs)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported input admitted')
        assert not events
    wrong_device,wrong_events=host(current_device=0)
    try:wrong_device(*base)
    except ValueError:pass
    else:raise AssertionError('ambient CUDA device mismatch admitted')
    assert not wrong_events
    original,_=host(ORIGINAL)
    # Original shares cosine's row stride with sine and reads the wrong row.
    q=Tensor(torch.ones(2,1,4));k=Tensor(torch.ones(2,1,4))
    c=Tensor(torch.zeros(2,2));sine=Tensor(torch.tensor([[1.,2.,71.],[3.,4.,73.]]))
    expected=reference(q.data.clone(),c.data,sine.data)
    original(q,k,c,sine)
    assert not torch.equal(q.data,expected)
    # Original cache GQA writes inactive K heads into a different cache block.
    q=Tensor(torch.ones(1,4,4));raw_k=torch.full((16,),7.)
    k=Tensor(raw_k[:8].reshape(1,2,4));c=Tensor(torch.ones(1,2));sine=Tensor(torch.zeros(1,2))
    cache=Tensor(torch.full((2,2,4,4),-97.))
    original(q,k,c,sine,k_cache=cache,block_tables=Tensor(torch.zeros(1,1,dtype=torch.int64)),
        kv_lengths=Tensor(torch.ones(1,dtype=torch.int64)))
    assert torch.equal(cache.data[1,:,0,:],torch.full((2,4),7.))
    print('Original independent-sine-stride and cached-GQA corruption witnesses reproduced.')
    print(f'{paths} complete no-cache/cache CPU source paths; {len(failures)+1} prewrite rejections.')
    print('Q/K identity, cache K preservation, selected slots and byte tails checked. CUDA/numerics not tested.')

if __name__=='__main__':main()
