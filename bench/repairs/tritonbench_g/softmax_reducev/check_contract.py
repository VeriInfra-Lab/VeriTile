"""CPU structural controls for the online source; no GPU numerical claim."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'softmax_reducev_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/softmax_reducev/softmax_reducev.py'


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_checks():
    original = function(ORIGINAL, '_fwd_kernel')
    repaired = function(SOURCE, '_fwd_kernel')
    loop = next(n for n in original.body if isinstance(n, ast.For))
    other = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'v_index')
    next(k for k in other.value.keywords if k.arg == 'other').value = ast.Constant(0)
    vload = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'v')
    vload.value.keywords = ast.parse('v = tl.load(v_ptrs + v_index[:, None] * stride_vbs, mask=(start_n + offs_n < cur_batch_seq_len)[:, None], other=0.0)').body[0].value.keywords
    assert ast.dump(original) == ast.dump(repaired)
    wrapper = ast.unparse(function(SOURCE, 'token_softmax_reducev_fwd'))
    assert all(text not in wrapper for text in ('.clone(', '.contiguous(', '.cpu(', '.item(', '.reshape('))
    assert 'num_stages=3' in wrapper and 'num_warps=num_warps' in wrapper
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()


class Tensor:
    def __init__(self, data=None, *, shape=None, dtype=torch.float32, cuda=True,
                 layout=torch.strided, contiguous=True, negative=False, pointer=4096):
        self.data = data
        self.shape = tuple(data.shape) if data is not None else tuple(shape)
        self.dtype = data.dtype if data is not None else dtype
        self.layout, self.is_cuda, self.device = layout, cuda, torch.device('cuda:0')
        self._contiguous, self._negative, self._pointer = contiguous, negative, pointer
    def __getitem__(self, index): return self.data[index]
    @property
    def ndim(self): return len(self.shape)
    def numel(self):
        from math import prod
        return prod(self.shape)
    def is_contiguous(self): return self.data.is_contiguous() if self.data is not None else self._contiguous
    def is_neg(self): return self.data.is_neg() if self.data is not None else self._negative
    def data_ptr(self):
        assert self.numel(), 'no empty pointer query'
        return self.data.data_ptr() if self.data is not None else self._pointer
    def untyped_storage(self): return self.data.untyped_storage()


def raw(t):
    return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes()//4,),(1,))


class Pointer:
    def __init__(self,tensor,offset=0):
        self.tensor,self.offset=tensor,torch.as_tensor(offset,dtype=torch.int64)
    def __add__(self,offset): return Pointer(self.tensor,self.offset+offset)


class Kernel:
    def __init__(self):
        self.calls,self.pids=[],(0,0)
        def load(p,mask=None,other=None):
            if mask is None: return raw(p.tensor)[p.tensor.storage_offset()+p.offset]
            mask=torch.broadcast_to(mask,p.offset.shape)
            result=torch.full_like(p.offset,other,dtype=p.tensor.dtype)
            result[mask]=raw(p.tensor)[p.tensor.storage_offset()+p.offset[mask]]
            return result
        def store(p,value): raw(p.tensor)[p.tensor.storage_offset()+p.offset]=value
        tl=SimpleNamespace(program_id=lambda d:torch.tensor(self.pids[d],dtype=torch.int32),
            load=load,store=store,arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),
            zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),float32=torch.float32,
            multiple_of=lambda x,_:x,exp=lambda x:torch.exp(torch.as_tensor(x)),
            maximum=lambda x,y:torch.maximum(torch.as_tensor(x),torch.as_tensor(y)),
            max=lambda x,axis:x.max(dim=axis).values,sum=lambda x,axis:x.sum(dim=axis))
        node=function(SOURCE,'_fwd_kernel');node.decorator_list=[]
        for a in node.args.args:a.annotation=None
        env={'tl':tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node],[])),str(SOURCE),'exec'),env)
        self.body=env[node.name]
    def __getitem__(self,grid):
        def launch(x,v,o,p,s,n,maximum,*rest,BLOCK_DMODEL,BLOCK_N,num_warps,num_stages):
            assert BLOCK_N==64 and num_warps==1 and num_stages==3
            self.calls.append((grid,x,v,o,p,s,n,maximum,rest,BLOCK_DMODEL))
            for b in range(grid[0]):
                for h in range(grid[1]):
                    self.pids=b,h
                    self.body(*[Pointer(t.data) for t in (x,v,o,p,s,n)],maximum,*rest,BLOCK_DMODEL,BLOCK_N)
        return launch


def harness():
    kernel=Kernel();node=function(SOURCE,'token_softmax_reducev_fwd');node.decorator_list=[]
    content=function(SOURCE,'_check_contents')
    env={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int32=torch.int32,
            cuda=SimpleNamespace(device=lambda _:nullcontext())), '_fwd_kernel':kernel}
    exec(compile(ast.fix_missing_locations(ast.Module([content,node],[])),str(SOURCE),'exec'),env)
    return env[node.name],kernel


def checked_case(dim,count,offset=False):
    call,kernel=harness();heads,batch,maximum=2,2,count+3
    xbase=torch.randn(heads*(2*count+4)+7);x=xbase[3:3+heads*(2*count+4)].reshape(heads,2*count+4)
    vbase=torch.randn((7*heads*dim)+9);v=vbase[4:4+7*heads*dim].reshape(7,heads,dim)
    v[0]=float('nan')
    obase=torch.full((batch*heads*dim+9,),91.);o=obase[2:2+batch*heads*dim].reshape(batch,heads,dim)
    p=torch.full((batch,maximum),-999,dtype=torch.int32)
    lengths=torch.tensor([count,max(1,count-1)],dtype=torch.int32)
    starts=torch.tensor([0,count+3],dtype=torch.int32)
    for b,n in enumerate(lengths.tolist()):p[b,maximum-n:]=torch.arange(n)%6+1
    if offset:
        p=torch.cat([torch.tensor([-42],dtype=torch.int32),p.reshape(-1),torch.tensor([-43],dtype=torch.int32)])[1:-1].reshape(p.shape)
        starts=torch.cat([torch.tensor([-42],dtype=torch.int32),starts,torch.tensor([-43],dtype=torch.int32)])[1:-1]
        lengths=torch.cat([torch.tensor([-42],dtype=torch.int32),lengths,torch.tensor([-43],dtype=torch.int32)])[1:-1]
    saved=[bytes(t.untyped_storage()) for t in (x,v,p,starts,lengths)]
    before=bytes(o.untyped_storage());objects=list(map(Tensor,(x,v,o,p,starts,lengths)))
    assert call(*objects,maximum,-2**50) is None
    expected=torch.empty_like(o)
    for b,n in enumerate(lengths.tolist()):
        expected[b]=torch.einsum('hk,khd->hd',torch.softmax(x[:,starts[b]:starts[b]+n],dim=1),v[p[b,maximum-n:].long()])
    torch.testing.assert_close(o,expected,rtol=4e-6,atol=3e-7)
    assert torch.isfinite(o).all()
    for t,b in zip((x,v,p,starts,lengths),saved):assert bytes(t.untyped_storage())==b
    after=bytes(o.untyped_storage());lo=4*o.storage_offset();hi=lo+4*o.numel()
    assert after[:lo]==before[:lo] and after[hi:]==before[hi:]
    event=kernel.calls[0];assert len(kernel.calls)==1 and event[0]==(batch,heads)
    assert all(event[i+1] is objects[i] for i in range(6))
    assert event[8]==(x.shape[1],1,heads*dim,dim,1,heads*dim,dim,1,maximum,1,-2**50)


def host_checks():
    count=0
    for d in (1,2,4,8,16,32,64,128):
        for n in (1,65):checked_case(d,n,offset=n==65);count+=1
    call,kernel=harness()
    for b,h in ((0,2),(2,0),(0,0)):
        ts=[Tensor(torch.empty(h,5)),Tensor(torch.empty(2,h,4)),Tensor(torch.empty(b,h,4)),
            Tensor(torch.empty(b,5,dtype=torch.int32)),Tensor(torch.empty(b,dtype=torch.int32)),
            Tensor(torch.empty(b,dtype=torch.int32))]
        assert call(*ts,5,-99) is None
        count+=1
    assert not kernel.calls
    def args():
        return [Tensor(torch.randn(2,5)),Tensor(torch.randn(3,2,4)),Tensor(torch.empty(1,2,4)),
            Tensor(torch.zeros(1,5,dtype=torch.int32)),Tensor(torch.zeros(1,dtype=torch.int32)),
            Tensor(torch.ones(1,dtype=torch.int32)),5,-9]
    controls=[(i,None) for i in range(6)]
    controls += [(0,Tensor(shape=(2,5),cuda=False)),(0,Tensor(shape=(2,5),layout=torch.sparse_coo)),
        (0,Tensor(torch.ones(2,5,dtype=torch.float16))),(3,Tensor(torch.zeros(1,5,dtype=torch.int64))),
        (0,Tensor(torch.ones(5,2).T)),(0,Tensor(shape=(2,5),negative=True)),
        (0,Tensor(shape=(10,))),(1,Tensor(torch.empty(3,2,3))),
        (2,Tensor(torch.empty(2,2,4))),(3,Tensor(torch.zeros(1,4,dtype=torch.int32))),
        (4,Tensor(torch.zeros(2,dtype=torch.int32))),(6,0),(6,-1),(6,True),(6,1.5),
        (0,Tensor(shape=(2,5),pointer=4097))]
    for index,value in controls:
        a=args();a[index]=value
        try:call(*a)
        except (TypeError,ValueError):pass
        else:raise AssertionError(('admitted control',index,value))
    def symbolic(batch, heads, dims, maximum=1, tokens=1, pages=1):
        return [Tensor(shape=(heads,tokens)),Tensor(shape=(pages,heads,dims)),Tensor(shape=(batch,heads,dims)),
            Tensor(shape=(batch,maximum),dtype=torch.int32),Tensor(shape=(batch,),dtype=torch.int32),
            Tensor(shape=(batch,),dtype=torch.int32),maximum,-9]
    structural=[symbolic(1,2**16,1),symbolic(1,1,2**15),symbolic(2**31,1,1),
        symbolic(1,2,4,pages=2**30),symbolic(1,1,1,maximum=2**31),symbolic(0,1,1,maximum=2**31)]
    for a in structural:
        try:call(*a)
        except (TypeError,ValueError):pass
        else:raise AssertionError('admitted structural limit')
    a=args();a[2]=Tensor(a[0].data.reshape(-1)[:8].reshape(1,2,4))
    try:call(*a)
    except ValueError:pass
    else:raise AssertionError('admitted aliased output')
    content_controls = [(4,[-1]),(4,[5]),(5,[-1]),(5,[0]),(5,[6]),
                        (3,[[-1,-1,-1,-1,-1]]),(3,[[0,0,0,0,3]])]
    for index, values in content_controls:
        a=args();a[index]=Tensor(torch.tensor(values,dtype=torch.int32))
        saved=bytes(a[2].data.untyped_storage())
        try:call(*a)
        except ValueError:pass
        else:raise AssertionError(('admitted invalid metadata',index,values))
        assert bytes(a[2].data.untyped_storage())==saved
    assert not kernel.calls
    print(f'{count} admission/online-loop controls and {len(controls)+len(structural)+1+len(content_controls)} rejections passed; padding NaNs and invalid sentinel are inert')


if __name__=='__main__':
    source_checks();host_checks()
