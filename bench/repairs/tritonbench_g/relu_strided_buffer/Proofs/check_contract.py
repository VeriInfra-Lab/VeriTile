"""Source-AST, signed-view guard and raw-word CPU interpretation checks.

CUDA device guard and native launch are mocked. The native kernel is unchanged.
"""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent.parent
SOURCE=HERE/'relu_strided_buffer_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/relu_strided_buffer/relu_strided_buffer.py'

class Tensor:
    def __init__(self,data,device='cuda:0',storage=None,pointer=None):
        self.data,self.device,self.storage,self.pointer=data,torch.device(device),storage,pointer
    @property
    def shape(self):return self.data.shape
    @property
    def dtype(self):return self.data.dtype
    @property
    def layout(self):return self.data.layout
    def stride(self):return self.data.stride()
    def numel(self):return self.data.numel()
    def is_neg(self):return self.data.is_neg()
    def untyped_storage(self):return self.data.untyped_storage() if self.storage is None else self.storage
    def data_ptr(self):return self.data.data_ptr() if self.pointer is None else self.pointer


def raw(data):
    return torch.empty(0,dtype=torch.float32).set_(data.untyped_storage(),0,
        (data.untyped_storage().nbytes()//4,),(1,))


def main():
    original={n.name:n for n in ast.parse(ORIGINAL.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    revised={n.name:n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in ['StridedBuffer','heuristics_for_tile_size','heuristics_for_num_warps',
                 'relu_forward','relu_forward_kernel_rank_1']:
        assert ast.dump(original[name])==ast.dump(revised[name]),name
    oldbody=original['relu_forward_wrapper_rank_1'].body[1:]
    newbody=revised['relu_forward_wrapper_rank_1'].body[-len(oldbody):]
    assert ast.dump(ast.Module(body=oldbody,type_ignores=[]))==ast.dump(ast.Module(body=newbody,type_ignores=[]))
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    spec=importlib.util.spec_from_file_location('relu_revision',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    SB=module.StridedBuffer
    events=[];launches=[];interpret=[True];pid=[0];ctas=[1]
    @contextlib.contextmanager
    def guard(index):
        events.append(('guard',index));yield
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        cuda=SimpleNamespace(_DeviceGuard=guard))
    def base(x):return x.unwrap() if isinstance(x,SB) else x
    class Pointer:
        type=SimpleNamespace(element_ty=torch.float32)
        def __init__(self,x):
            self.data=base(x).data
            self.offset=(x.data_ptr()-self.data.untyped_storage().data_ptr())//4
    class Block:
        type=SimpleNamespace(element_ty=torch.float32)
        def __init__(self,p,shape,strides,offsets,block_shape,order):
            self.p=p;self.indices=offsets[0]+torch.arange(block_shape[0]);self.mask=self.indices<shape[0]
            self.address=p.offset+self.indices*strides[0]
    def load(block,boundary_check):
        value=torch.full(block.indices.shape,float('nan'),dtype=torch.float32)
        value[block.mask]=raw(block.p.data)[block.address[block.mask]]
        return value
    def store(block,value,boundary_check):
        raw(block.p.data)[block.address[block.mask]]=value[block.mask]
    nodes=[revised['relu_forward'],revised['relu_forward_kernel_rank_1']]
    for node in nodes:
        node.decorator_list=[]
        for a in node.args.args:a.annotation=None
    ns={'tl':SimpleNamespace(program_id=lambda _:pid[0],num_programs=lambda _:ctas[0],
        cdiv=lambda n,b:(n+b-1)//b,make_block_ptr=Block,load=load,store=store,where=torch.where)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
    class Launch:
        def __getitem__(self,grid):
            def run(x,y,sx,orderx,sy,ordery,n,tasks,*,tiles_per_cta,tile_size0,one_tile_per_cta,num_warps):
                launches.append((grid,n,tiles_per_cta,tile_size0,one_tile_per_cta,num_warps))
                events.append(('launch',grid))
                if interpret[0]:
                    ctas[0]=grid[0]
                    for p in range(grid[0]):
                        pid[0]=p
                        ns['relu_forward_kernel_rank_1'](Pointer(x),Pointer(y),sx,orderx,sy,ordery,
                            n,tasks,tiles_per_cta,tile_size0,one_tile_per_cta)
            return run
    module.relu_forward_kernel_rank_1=Launch()
    def wrap(values):return Tensor(torch.tensor(values,dtype=torch.float32))
    def sb(t,n,s,offset=0):return SB(t,shape=(n,),strides=(s,),offset=offset)
    words=torch.tensor([0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x7f800000,
                        0xff800000,0x7f800001,0x7fc00001,0xffc00001],dtype=torch.int64).to(torch.int32)
    shared=Tensor(torch.arange(-10.,10.));a=Tensor(torch.arange(-10.,10.))
    cases=[(wrap([-2.,1.,-0.,3.,-5.]),wrap([91.]*5)),
        (sb(a,5,-1,8),wrap([91.]*5)),(wrap([-2.,1.,0.,3.,-5.]),sb(a,5,-1,8)),
        (sb(wrap([-2.,1.,0.,3.,-5.]),5,-1,4),sb(wrap([91.]*5),5,-1,4)),
        (sb(wrap([-2.]),5,0),wrap([91.]*5)),
        (sb(shared,10,2),sb(shared,10,2,1)),
        (sb(shared,5,-2,8),sb(shared,5,-2,9)),
        (sb(shared,5,1),sb(shared,5,1,10)),
        (Tensor(torch.arange(-10.,10.)[2:12:2]),wrap([91.]*5)),
        (Tensor(words.view(torch.float32)),wrap([91.]*len(words))),
        (Tensor(torch.arange(-512.,513.)),wrap([91.]*1025)),
        (sb(wrap([1.]),1,-(1<<63)),sb(wrap([91.]),1,(1<<63)-1))]
    def indices(x):return (x.data_ptr()-base(x).data.untyped_storage().data_ptr())//4+torch.arange(x.numel())*x.stride()[0]
    def expected(x):
        w=raw(base(x).data)[indices(x)].view(torch.int32).to(torch.int64)&0xffffffff
        nan=((w>>23)&255)==255
        nan=nan&((w&0x7fffff)!=0)
        keep=((w>>31)==0)&((w&0x7fffffff)!=0)&(~nan)
        return torch.where(keep,w,0).to(torch.int32)
    def run_case(x,y,force_loop=False):
        want=expected(x).clone()
        snapshots={base(z).data.untyped_storage().data_ptr():raw(base(z).data).view(torch.int32).clone() for z in [x,y]}
        target=base(y).data.untyped_storage().data_ptr();outidx=indices(y)
        events.clear()
        if force_loop:
            # Exercise the unchanged loop AST on a smaller explicit two-CTA launch.
            ctas[0]=2
            for p in range(2):
                pid[0]=p;ns['relu_forward_kernel_rank_1'](Pointer(x),Pointer(y),x.stride()[0],0,
                    y.stride()[0],0,x.numel(),x.numel(),2,512,False)
        else:
            assert module.relu_forward_wrapper_rank_1(x,out0=y) is y
            assert [e[0] for e in events]==['guard','launch']
        assert torch.equal(raw(base(y).data)[outidx].view(torch.int32),want)
        for z in [x,y]:
            storage=base(z).data.untyped_storage().data_ptr();before=snapshots[storage]
            after=raw(base(z).data).view(torch.int32)
            outside=torch.ones(len(before),dtype=torch.bool)
            if storage==target:outside[outidx]=False
            assert torch.equal(before[outside],after[outside])
    for x,y in cases:run_case(x,y)
    run_case(Tensor(torch.arange(-512.,513.)),wrap([91.]*1025),True)
    combinations=0
    for n in range(1,8):
        for sx in range(-3,4):
            for sy in [-3,-2,-1,1,2,3]:
                for a in range(-4,5):
                    for b in range(-4,5):
                        want=set(a+i*sx for i in range(n)).isdisjoint(b+j*sy for j in range(n))
                        assert module._footprints_disjoint(a,sx,b,sy,n)==want
                        combinations+=1
    valid=wrap([1.,2.,3.]);out=wrap([91.]*3)
    malformed=sb(valid,3,1);malformed._strides=(True,)
    pointer=sb(valid,3,1);pointer._data_ptr+=1
    broken_base=sb(valid,3,1);broken_base._base=[]
    broken_pointer=sb(valid,3,1);broken_pointer._data_ptr='bad'
    wrong_device=sb(valid,3,1);wrong_device.device=torch.device('cuda:1')
    wrong_dtype=sb(valid,3,1);wrong_dtype.dtype=torch.float16
    odd_storage=torch.arange(20,dtype=torch.uint8).untyped_storage()[1:17]
    unaligned=Tensor(torch.empty(0,dtype=torch.float32).set_(odd_storage,0,(3,),(1,)))
    bad=[(broken_base,out),(broken_pointer,out),(wrong_device,out),(wrong_dtype,out),(unaligned,out),([],out),(Tensor(torch.eye(3).to_sparse()),out),(Tensor(torch.ones(3,dtype=torch.float16)),out),
        (Tensor(torch.ones(3),'cpu'),out),(Tensor(torch._neg_view(torch.ones(3))),out),
        (Tensor(torch.ones(1,3)),out),(Tensor(torch.empty(0)),Tensor(torch.empty(0))),
        (Tensor(torch.empty(1<<31,device='meta')),out),(sb(valid,3,1<<63),out),(sb(valid,3,-(1<<63)-1),out),
        (pointer,out),(sb(valid,3,1,1),out),(sb(valid,3,-1,0),out),(valid,wrap([91.,91.])),
        (valid,Tensor(torch.ones(3),'cuda:1')),(valid,sb(out,3,0)),(valid,valid),(malformed,out),
        (sb(valid,-3,1),out),(sb(valid,3.0,1),out)]
    for x,y in bad:
        events.clear()
        try:module.relu_forward_wrapper_rank_1(x,out0=y)
        except (ValueError,TypeError):pass
        else:raise AssertionError('unsupported view admitted')
        assert events==[]
    # Host-only capped-grid endpoints, without allocating or executing giant CPU tensors.
    interpret[0]=False
    for n in [33554432,33554433,(1<<31)-1]:
        def huge(start):
            storage=SimpleNamespace(data_ptr=lambda:start,nbytes=lambda:4*n)
            return Tensor(torch.empty(n,device='meta'),storage=storage,pointer=start)
        x,y=huge(4096),huge(1<<40);events.clear()
        assert module.relu_forward_wrapper_rank_1(x,out0=y) is y
        grid,nn,t,b,one,warps=launches[-1]
        assert grid==(65536,1,1) and b==512 and one==(n==33554432) and warps==4
        assert t==((n+511)//512+65535)//65536
    print(f'{len(cases)} raw-word output/frame wrapper cases, one direct loop-AST case, {len(bad)} rejection cases passed.')
    print(f'{combinations} exact signed footprint checks and three capped-grid host endpoints passed.')
    print('Source identity passed; CUDA allocation/launch were mocked. No GPU execution claim.')

if __name__=='__main__':main()
