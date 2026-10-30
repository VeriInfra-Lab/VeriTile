"""Accepted signed-control JIT AST on CPU backing storage; CUDA is mocked."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'kcache_copy_triton_repaired.py'
NAME='_copy_to_kcache_seqlen_n_kernel_repaired'
spec=importlib.util.spec_from_file_location('copy_helpers',HERE.parent/'destindex_copy_kv1/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,backing_bytes,Pointer=helpers.function,helpers.raw,helpers.backing_bytes,helpers.Pointer
ACCEPTED_AST_SHA256='49ce2df31b9f4741aa5732672b1253956653bb0961fdefd85f29370c10dec068'

class Tensor(helpers.Tensor):
    def reshape(self,*shape):return Tensor(self.data.reshape(*shape),str(self.device),self.is_cuda)


def source_binding():
    assert hashlib.sha256(ast.dump(function(SOURCE,NAME),include_attributes=False).encode()).hexdigest()==ACCEPTED_AST_SHA256
    original=function(SOURCE,NAME)
    candidate=function(SOURCE,'_copy_to_kcache_seqlen_n_kernel_commuted')
    candidate.name=NAME
    offsets=next(x for x in candidate.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='offsets_k')
    assert isinstance(offsets.value,ast.BinOp) and isinstance(offsets.value.left,ast.BinOp)
    offsets.value.left.left,offsets.value.left.right=offsets.value.left.right,offsets.value.left.left
    assert ast.dump(original,include_attributes=False)==ast.dump(candidate,include_attributes=False)
    wrapper=function(SOURCE,'_checked_copy')
    forbidden={'clone','contiguous','resolve_neg','copy_','to','zeros','empty','tensor'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(wrapper))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='reshape' for n in ast.walk(wrapper))==1
    lean=(HERE/'Contract.lean').read_text().split('def checkInputs',1)[1].split('def checkArguments',1)[0]
    assert re.findall(r'\.error \.(\w+)',lean)==['n','layout','dtype','cuda','device','rank','contiguous','negative','shapes','cacheShape','geometry','grid','extent','tile','alignment','overlap','bounds','unique']
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()


def interpreter(name):
    pid=[0,0,0]
    def load(pointer):
        offsets=torch.as_tensor(pointer.offset+pointer.data.storage_offset()).long()
        storage=raw(pointer.data)
        assert bool(((offsets>=0)&(offsets<len(storage))).all()),'source OOB'
        return storage[offsets]
    def store(pointer,value):
        offsets=torch.as_tensor(pointer.offset+pointer.data.storage_offset()).long()
        storage=raw(pointer.data)
        assert bool(((offsets>=0)&(offsets<len(storage))).all()),'output OOB'
        storage[offsets]=value
    node=function(SOURCE,name);node.decorator_list=[]
    for a in node.args.args:a.annotation=None
    ns={'tl':SimpleNamespace(program_id=lambda axis:torch.tensor(pid[axis],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),load=load,store=store,int64=torch.int64)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[name],pid


def main(variant):
    source_binding()
    spec=importlib.util.spec_from_file_location('direct_kcache',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    name = NAME if variant == 'repaired' else '_copy_to_kcache_seqlen_n_kernel_commuted'
    kernel,pid=interpreter(name);calls=[]
    class Launch:
        def __getitem__(self,grid):
            def launch(k,cache,table,lengths,*args,**kw):
                calls.append((grid,k,cache,table,lengths,args,kw))
                for s in reversed(range(grid[0])):
                    for head in reversed(range(grid[1])):
                        for split in reversed(range(grid[2])):
                            pid[:]=[s,head,split]
                            kernel(*(Pointer(t.data) for t in (k,cache,table,lengths)),*args,kw['n_tokens'],kw['HEAD_DIM'],kw['KCACHE_X'])
            return launch
    setattr(module,name,Launch())
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int32=torch.int32,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    run=getattr(module,'copy_k_to_blocked_cache_'+variant)
    cases=[(0,1,3,8,8,False,False),(1,1,1,1,1,False,False),(2,2,3,8,8,False,True),
        (3,3,2,16,16,False,False),(2,5,3,6,2,True,False),(3,2,5,24,8,True,True),(1,7,2,256,16,True,True)]
    for sequences,n,heads,dims,width,new,rank4 in cases:
        count=sequences*n;size=count*heads*dims
        bits=torch.arange(size+8,dtype=torch.int32)*77123
        bits[:6]=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1],dtype=torch.int32)
        k=bits.view(torch.float32)[1:1+size].reshape(count,heads,dims)
        block_size,columns=4,4+(n+sequences)//4
        blocks=sequences*columns+1
        out_size=blocks*heads*block_size*dims
        cache=torch.full((out_size+8,),41.)[2:2+out_size]
        cache=cache.reshape(blocks,heads,dims//width,block_size,width) if new else cache.reshape(blocks,heads,block_size,dims)
        lengths=torch.arange(sequences,dtype=torch.int32)+block_size+n
        tables=torch.arange(sequences*columns,dtype=torch.int32).reshape(sequences,columns)
        expected=cache.view(torch.int32).clone()
        for token in range(count):
            seq,u=divmod(token,n);pos=int(lengths[seq])-n+u
            block=int(tables[seq,pos//block_size]);slot=pos%block_size
            if new:expected[block,:,:,slot,:]=k.view(torch.int32)[token].reshape(heads,dims//width,width)
            else:expected[block,:,slot,:]=k.view(torch.int32)[token]
        data=(k.reshape(sequences,n,heads,dims) if rank4 else k,cache,lengths,tables)
        args=[Tensor(x) for x in data];snapshots=[backing_bytes(x).clone() for x in data]
        old=len(calls);assert run(*args,n=n,use_new_kcache_layout=new) is None
        assert torch.equal(cache.view(torch.int32),expected)
        for i,(x,snapshot) in enumerate(zip(data,snapshots)):
            if i!=1:assert torch.equal(backing_bytes(x),snapshot)
            start=x.storage_offset()*4;end=start+x.numel()*4
            assert torch.equal(backing_bytes(x)[:start],snapshot[:start]) and torch.equal(backing_bytes(x)[end:],snapshot[end:])
        assert len(calls)-old==int(count>0)
        if count:
            assert calls[-1][2] is args[1]
            assert calls[-1][1].data_ptr()==args[0].data_ptr() and calls[-1][1].shape==(count,heads,dims)
    def fresh():
        return [Tensor(torch.ones(4,2,8)),Tensor(torch.zeros(6,2,4,8)),Tensor(torch.tensor([3,4],dtype=torch.int32)),Tensor(torch.tensor([[0,1],[2,3]],dtype=torch.int32))]
    rejected=0
    def reject(args,n=2,new=False):
        nonlocal rejected
        old=len(calls);snapshots=[backing_bytes(x.data).clone() for x in args if isinstance(x,Tensor)]
        try:run(*args,n=n,use_new_kcache_layout=new)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported request admitted')
        assert len(calls)==old
        assert all(torch.equal(s,backing_bytes(x.data)) for s,x in zip(snapshots,[x for x in args if isinstance(x,Tensor)]))
        rejected+=1
    a=fresh();a[0]=object();reject(a)
    for n in [None,True,1.5,0,-1,1<<31]:reject(fresh(),n=n)
    reject(fresh(),new=1)
    class WrongLayout(Tensor):
        @property
        def layout(self):return torch.sparse_coo
    a=fresh();a[0]=WrongLayout(a[0].data);reject(a)
    for i in range(4):
        a=fresh();a[i].is_cuda=False;reject(a)
        a=fresh();a[i].device=torch.device('cuda:1');reject(a)
        a=fresh();a[i]=Tensor(a[i].data.to(torch.float64));reject(a)
        a=fresh();a[i].is_contiguous=lambda:False;reject(a)
        a=fresh();a[i].is_neg=lambda:True;reject(a)
        a=fresh();a[i].numel=lambda:1<<63;reject(a)
        a=fresh();a[i].data_ptr=lambda:3;reject(a)
    for i,shape,dtype in [(0,(4,8),torch.float32),(1,(6,2,4,16),torch.float32),(2,(2,1),torch.int32),
        (3,(2,2,1),torch.int32),(2,(3,),torch.int32),(3,(3,2),torch.int32)]:
        a=fresh();a[i]=Tensor(torch.zeros(shape,dtype=dtype));reject(a)
    reject(fresh(),n=3)
    reject(fresh(),new=True)
    for i in range(4):
        for j in range(i+1,4):
            a=fresh();ptr=a[i].data_ptr();a[j].data_ptr=lambda ptr=ptr:ptr;reject(a)
    for lens in [[1,4],[-1,4],[9,4]]:
        a=fresh();a[2]=Tensor(torch.tensor(lens,dtype=torch.int32));reject(a)
    for blocks in [[[-1,1],[2,3]],[[6,1],[2,3]]]:
        a=fresh();a[3]=Tensor(torch.tensor(blocks,dtype=torch.int32));reject(a)
    a=fresh();a[2]=Tensor(torch.tensor([3,3],dtype=torch.int32));a[3]=Tensor(torch.tensor([[0,1],[0,2]],dtype=torch.int32));reject(a)
    for heads,dims,size in [(0,8,4),(2,0,4),(2,8,0),(2,3,4)]:
        a=[Tensor(torch.empty(0,heads,dims)),Tensor(torch.empty(3,heads,size,dims)),
            Tensor(torch.empty(0,dtype=torch.int32)),Tensor(torch.empty(0,2,dtype=torch.int32))]
        reject(a)
    for heads,dims in [(65536,8),(1,1<<21)]:
        a=[Tensor(torch.empty(0,heads,dims)),Tensor(torch.empty(0,heads,4,dims)),
            Tensor(torch.empty(0,dtype=torch.int32)),Tensor(torch.empty(0,2,dtype=torch.int32))]
        reject(a)
    a=[Tensor(torch.empty(0,1,65536)),Tensor(torch.empty(0,1,65536,4,1)),Tensor(torch.empty(0,dtype=torch.int32)),Tensor(torch.empty(0,2,dtype=torch.int32))]
    reject(a,new=True)
    # Historical peer003 launcher-overflow witness, represented without payload allocation.
    class Descriptor(Tensor):
        def __init__(self,data,shape):super().__init__(data);self.descriptor_shape=shape
        @property
        def shape(self):return self.descriptor_shape
        @property
        def ndim(self):return len(self.shape)
        def numel(self):
            result=1
            for size in self.shape:result*=size
            return result
    a=[Descriptor(torch.empty(0),(65536,32768,1)),Descriptor(torch.empty(0),(1,32768,65536,1)),
        Tensor(torch.tensor([65536],dtype=torch.int32)),Tensor(torch.tensor([[0]],dtype=torch.int32))]
    reject(a,n=65536)
    old_product=torch.tensor(65536,dtype=torch.int32)*32768
    assert int(old_product)==-(1<<31) and 65536*32768==1<<31
    # The original un-widened source pointer product wraps before pointer addition.
    assert int(torch.tensor(65536,dtype=torch.int32)*65536)==0
    assert int(torch.tensor(65536,dtype=torch.int64)*65536)==1<<32
    print(f'{variant}: {len(cases)} admissions and {rejected} rejections passed.')
    print('Both layouts, n-token block crossings, original reshape, raw FP32 bits, exterior bytes, int32 pointer overflow and historical grid overflow checked on CPU; CUDA mocked.')


if __name__=='__main__':
    main('repaired')
    main('commuted')
