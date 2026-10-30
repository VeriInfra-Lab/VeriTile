"""Original JIT AST on CPU backing storage; CUDA launch/context are mocked."""
import ast
import contextlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'kv_cache_copy_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/kv_cache_copy/kv_cache_copy.py'
NAME='_copy_to_kvcache_seqlen1_kernel'
spec=importlib.util.spec_from_file_location('copy_helpers',HERE.parent/'destindex_copy_kv1/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,backing_bytes,Pointer=helpers.function,helpers.raw,helpers.backing_bytes,helpers.Pointer

class Tensor(helpers.Tensor):
    def squeeze(self,dim):return Tensor(self.data.squeeze(dim),str(self.device),self.is_cuda)


def source_binding():
    assert ast.dump(function(SOURCE,NAME),include_attributes=False)==ast.dump(function(ORIGINAL,NAME),include_attributes=False)
    original=function(SOURCE,NAME)
    candidate=function(SOURCE,NAME+'_commuted');candidate.name=NAME
    loop=next(x for x in candidate.body if isinstance(x,ast.For))
    offset=next(x for x in loop.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='offsets_k')
    offset.value.left.left,offset.value.left.right=offset.value.left.right,offset.value.left.left
    assert ast.dump(original,include_attributes=False)==ast.dump(candidate,include_attributes=False)
    wrapper=function(SOURCE,'_checked_copy')
    forbidden={'clone','contiguous','resolve_neg','copy_','to','zeros','empty','tensor'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(wrapper))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='squeeze' for n in ast.walk(wrapper))==2
    lean=(HERE/'Contract.lean').read_text().split('def checkInputs',1)[1].split('def checkArguments',1)[0]
    assert re.findall(r'\.error \.(\w+)',lean)==['layout','dtype','cuda','device','rank','contiguous','negative','shapes','keyShape','geometry','grid','extent','tile','alignment','overlap','bounds','unique']
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()


def interpreter(name):
    pid=[0,0]
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
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),static_range=range,load=load,store=store)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[name],pid


def main(variant):
    source_binding()
    spec=importlib.util.spec_from_file_location('direct_kvcache',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    name=NAME if variant=='repaired' else NAME+'_commuted'
    kernel,pid=interpreter(name);calls=[]
    class Launch:
        def __getitem__(self,grid):
            def launch(k,v,kc,vc,table,lengths,*args,**kw):
                calls.append((grid,k,v,kc,vc,table,lengths,args,kw))
                for s in reversed(range(grid[0])):
                    for head in reversed(range(grid[1])):
                        pid[:]=[s,head]
                        kernel(*(Pointer(t.data) for t in (k,v,kc,vc,table,lengths)),*args,kw['HEAD_DIM'],kw['KCACHE_X'])
            return launch
    setattr(module,name,Launch())
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int32=torch.int32,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    run=getattr(module,'copy_kv_to_blocked_cache_'+variant)
    cases=[(0,3,8,8,False,False),(1,1,1,1,False,False),(2,3,8,8,False,True),
        (3,2,16,16,False,False),(2,3,6,2,True,False),(3,5,24,8,True,True),(1,2,256,16,True,False)]
    for count,heads,dims,width,new,rank4 in cases:
        size=count*heads*dims
        kb=torch.arange(size+8,dtype=torch.int32)*77123;vb=torch.arange(size+8,dtype=torch.int32)*-17957
        kb[:6]=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1],dtype=torch.int32)
        k=kb.view(torch.float32)[1:1+size].reshape(count,heads,dims)
        v=vb.view(torch.float32)[2:2+size].reshape(count,heads,dims)
        blocks,block_size=count+3,4
        out_size=blocks*heads*block_size*dims
        kc=torch.full((out_size+8,),41.)[2:2+out_size]
        vc=torch.full((out_size+8,),43.)[3:3+out_size]
        kc=kc.reshape(blocks,heads,dims//width,block_size,width) if new else kc.reshape(blocks,heads,block_size,dims)
        vc=vc.reshape(blocks,heads,block_size,dims)
        lengths=torch.arange(count,dtype=torch.int32)%block_size+block_size+1
        tables=torch.full((count,3),-7,dtype=torch.int32)
        if count:tables[:,1]=torch.arange(count,dtype=torch.int32)
        expected_k,expected_v=kc.view(torch.int32).clone(),vc.view(torch.int32).clone()
        for s in range(count):
            pos=int(lengths[s])-1;block=int(tables[s,pos//block_size]);slot=pos%block_size
            if new:expected_k[block,:,:,slot,:]=k.view(torch.int32)[s].reshape(heads,dims//width,width)
            else:expected_k[block,:,slot,:]=k.view(torch.int32)[s]
            expected_v[block,:,slot,:]=v.view(torch.int32)[s]
        data=(k.unsqueeze(1) if rank4 else k,v.unsqueeze(1) if rank4 else v,kc,vc,lengths,tables)
        args=[Tensor(x) for x in data]
        snapshots=[backing_bytes(x).clone() for x in data]
        old=len(calls);assert run(*args,use_new_kcache_layout=new) is None
        assert torch.equal(kc.view(torch.int32),expected_k) and torch.equal(vc.view(torch.int32),expected_v)
        for i,(x,snapshot) in enumerate(zip(data,snapshots)):
            if i not in (2,3):assert torch.equal(backing_bytes(x),snapshot)
            start=x.storage_offset()*4;end=start+x.numel()*4
            assert torch.equal(backing_bytes(x)[:start],snapshot[:start]) and torch.equal(backing_bytes(x)[end:],snapshot[end:])
        assert len(calls)-old==int(count>0)
        if count:
            assert calls[-1][3] is args[2] and calls[-1][4] is args[3]
            assert calls[-1][1].data_ptr()==args[0].data_ptr() and calls[-1][2].data_ptr()==args[1].data_ptr()
    # Different original ranks normalize to the same source geometry.
    def fresh():
        return [Tensor(torch.ones(2,2,8)),Tensor(torch.ones(2,1,2,8)),Tensor(torch.zeros(3,2,4,8)),
            Tensor(torch.zeros(3,2,4,8)),Tensor(torch.tensor([1,2],dtype=torch.int32)),Tensor(torch.tensor([[0,1],[1,2]],dtype=torch.int32))]
    assert run(*fresh()) is None
    rejected=0
    def reject(args,new=False):
        nonlocal rejected
        old=len(calls);snapshots=[backing_bytes(x.data).clone() for x in args if isinstance(x,Tensor)]
        try:run(*args,use_new_kcache_layout=new)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported request admitted')
        assert len(calls)==old
        assert all(torch.equal(s,backing_bytes(x.data)) for s,x in zip(snapshots,[x for x in args if isinstance(x,Tensor)]))
        rejected+=1
    a=fresh();a[0]=object();reject(a)
    reject(fresh(),1)
    class WrongLayout(Tensor):
        @property
        def layout(self):return torch.sparse_coo
    a=fresh();a[0]=WrongLayout(a[0].data);reject(a)
    for i in range(6):
        a=fresh();a[i].is_cuda=False;reject(a)
        a=fresh();a[i].device=torch.device('cuda:1');reject(a)
        a=fresh();a[i]=Tensor(a[i].data.to(torch.float64));reject(a)
        a=fresh();a[i].is_contiguous=lambda:False;reject(a)
        a=fresh();a[i].is_neg=lambda:True;reject(a)
        a=fresh();a[i].numel=lambda:1<<31;reject(a)
        a=fresh();a[i].data_ptr=lambda:3;reject(a)
    for i in (0,1):
        a=fresh();a[i]=Tensor(torch.ones(2,8));reject(a)
        a=fresh();a[i]=Tensor(torch.ones(2,2,2,8));reject(a)
    for i,shape,dtype in [(1,(2,2,16),torch.float32),(2,(3,2,4,16),torch.float32),(3,(3,3,4,8),torch.float32),
        (4,(2,1),torch.int32),(5,(2,2,1),torch.int32),(4,(3,),torch.int32),(5,(3,2),torch.int32)]:
        a=fresh();a[i]=Tensor(torch.zeros(shape,dtype=dtype));reject(a)
    reject(fresh(),True)
    for i in range(6):
        for j in range(i+1,6):
            a=fresh();ptr=a[i].data_ptr();a[j].data_ptr=lambda ptr=ptr:ptr;reject(a)
    for lens in [[0,2],[-1,2],[9,2]]:
        a=fresh();a[4]=Tensor(torch.tensor(lens,dtype=torch.int32));reject(a)
    for blocks in [[[-1,1],[1,2]],[[3,1],[1,2]]]:
        a=fresh();a[5]=Tensor(torch.tensor(blocks,dtype=torch.int32));reject(a)
    a=fresh();a[4]=Tensor(torch.tensor([1,1],dtype=torch.int32));a[5]=Tensor(torch.tensor([[0,1],[0,2]],dtype=torch.int32));reject(a)
    for heads,dims,size in [(0,8,4),(2,0,4),(2,8,0),(2,3,4)]:
        a=[Tensor(torch.empty(0,heads,dims)),Tensor(torch.empty(0,heads,dims)),Tensor(torch.empty(3,heads,size,dims)),
            Tensor(torch.empty(3,heads,size,dims)),Tensor(torch.empty(0,dtype=torch.int32)),Tensor(torch.empty(0,2,dtype=torch.int32))]
        reject(a)
    # Empty tensors permit resource-bound guard cases without huge allocations.
    for heads,dims in [(65536,8),(1,1<<21),(1,1<<31)]:
        a=[Tensor(torch.empty(0,heads,dims)),Tensor(torch.empty(0,heads,dims)),Tensor(torch.empty(0,heads,4,dims)),
            Tensor(torch.empty(0,heads,4,dims)),Tensor(torch.empty(0,dtype=torch.int32)),Tensor(torch.empty(0,2,dtype=torch.int32))]
        reject(a)
    print(f'{variant}: original kernel AST preserved; {len(cases)+1} admissions and {rejected} rejections passed.')
    print('Both layouts, full static loops, original squeeze views, FP32 raw bits and exterior backing bytes checked on CPU; CUDA mocked.')


def source_counterexamples():
    # The original wrapper admits this shape but the JIT ignores KCache's
    # final-axis stride. Our explicit contiguous contract rejects this input.
    kernel,pid=interpreter(NAME)
    k=torch.tensor([[[1.,2.]]]);v=torch.tensor([[[3.,4.]]])
    backing=torch.full((1,1,2,4),-1.)
    kc=backing[...,::2];vc=torch.full((1,1,2,2),-1.)
    lengths=torch.tensor([1],dtype=torch.int32);table=torch.tensor([[0]],dtype=torch.int32)
    kernel(*(Pointer(x) for x in (k,v,kc,vc,table,lengths)),
           *k.stride(),*v.stride(),kc.stride(0),kc.stride(1),0,kc.stride(2),kc.stride(3),
           *vc.stride(),*table.stride(),2,2,2)
    assert kc[0,0,0].tolist()==[1.,-1.]
    assert backing[0,0,0,1].item()==2.
    assert not torch.equal(kc[0,0,0],k[0,0])
    # Original INT32 offset arithmetic wraps outside the guarded extent.
    assert int(torch.tensor(65536,dtype=torch.int32)*65536)==0
    assert 65536*65536==1<<32
    # Exact shifted arange lowering used by the canonical DSL, including
    # non-power-of-two split counts, has the original lane order and dtype.
    for width in (1,2,8,16):
        for split in range(7):
            assert torch.equal(torch.arange(split*width,(split+1)*width,dtype=torch.int32),
                               split*width+torch.arange(width,dtype=torch.int32))
    print('Original noncontiguous-cache and INT32 offset-overflow counterexamples reproduced; shifted arange source lowering checked.')


if __name__=='__main__':
    main('repaired')
    main('commuted')
    source_counterexamples()
