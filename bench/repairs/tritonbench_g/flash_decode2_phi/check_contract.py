"""CPU execution of the actual guarded host and JIT AST; no GPU numerical claim."""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent


def raw(t):
    return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,
        (t.untyped_storage().nbytes()//t.element_size(),),(1,))


class Tensor:
    def __init__(self,data,*,device=0,cuda=True,layout=None,pointer=None,conjugate=False):
        self.data=data; self.device=torch.device(f"cuda:{device}"); self.is_cuda=cuda
        self.layout=data.layout if layout is None else layout
        self.dtype=data.dtype; self.shape=data.shape; self.ndim=data.ndim
        self.pointer=pointer; self.conjugate=conjugate; self.content_reads=0; self.pointer_reads=0
    def numel(self):return self.data.numel()
    def stride(self,*a):return self.data.stride(*a)
    def is_contiguous(self):return self.data.is_contiguous()
    def storage_offset(self):return self.data.storage_offset()
    def is_neg(self):return self.data.is_neg()
    def is_conj(self):return self.conjugate or self.data.is_conj()
    def data_ptr(self):
        self.pointer_reads+=1
        return self.data.data_ptr() if self.pointer is None else self.pointer
    def untyped_storage(self):return self.data.untyped_storage()
    def __getitem__(self,key):self.content_reads+=1;return self.data[key]


class Pointer:
    def __init__(self,tensor,offset=0):self.tensor=tensor;self.offset=torch.as_tensor(offset,dtype=torch.int64)
    def __add__(self,x):return Pointer(self.tensor,self.offset+x)


class Kernel:
    def __init__(self,node):
        self.pids=(0,0);self.events=[]
        def load(p,mask=None,other=None):
            index=p.offset+p.tensor.storage_offset();source=raw(p.tensor)
            if mask is None:return source[index]
            active=torch.broadcast_to(torch.as_tensor(mask),index.shape)
            out=torch.full(index.shape,other,dtype=p.tensor.dtype)
            out[active]=source[index[active]]
            return out
        def store(p,value,mask=None):
            index=p.offset+p.tensor.storage_offset();dest=raw(p.tensor)
            if mask is None:dest[index]=value
            else:dest[index[mask]]=value[mask]
        tl=SimpleNamespace(program_id=lambda a:torch.tensor(self.pids[a],dtype=torch.int32),
            int64=torch.int64,float32=torch.float32,arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),
            load=load,store=store,where=torch.where,zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),
            maximum=torch.maximum,exp=torch.exp)
        # tl.maximum accepts a scalar initial -infinity.
        tl.maximum=lambda a,b:torch.maximum(torch.as_tensor(a,dtype=torch.float32),torch.as_tensor(b,dtype=torch.float32))
        node.decorator_list=[]
        for a in node.args.args:a.annotation=None
        ns={"tl":tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node],[])),"flash_decode_kernel","exec"),ns)
        self.body=ns[node.name]
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            self.events.append((grid,args,kwargs.copy()))
            pointers=tuple(Pointer(x.data) if isinstance(x,Tensor) else x for x in args)
            kwargs.pop("num_warps");kwargs.pop("num_stages")
            for b in range(grid[0]):
                for h in range(grid[1]):self.pids=(b,h);self.body(*pointers,**kwargs)
        return launch


def harness(source):
    nodes=ast.parse(source.read_text()).body
    kernel_node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name.startswith("_fwd_kernel"))
    kernel=Kernel(kernel_node)
    context=SimpleNamespace(current=7,events=[])
    class Device:
        def __init__(self,device):self.device=device
        def __enter__(self):self.prior=context.current;context.current=self.device.index;context.events.append(self.device.index)
        def __exit__(self,*args):context.current=self.prior
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int32=torch.int32,
        cuda=SimpleNamespace(device=Device))
    host=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=="flash_decode_stage2")
    host.decorator_list=[]
    ns={"torch":fake,"triton":SimpleNamespace(next_power_of_2=lambda d:1<<(d-1).bit_length()),
        kernel_node.name:kernel}
    exec(compile(ast.fix_missing_locations(ast.Module([host],[])),str(source),"exec"),ns)
    return ns[host.name],kernel,context


def sample(B,H,S,D,lengths=None,Q=4):
    m=torch.randn(B,H,S,D);e=torch.randn(B,H,S)
    l=torch.tensor(lengths if lengths is not None else [max(1,S*Q-b) for b in range(B)],dtype=torch.int32)
    backing=torch.full((B*H*D+13,),12345.)
    o=backing[:B*H*D].view(B,H,D)
    return [Tensor(m),Tensor(e),Tensor(l),Tensor(o),Q],backing


def expected(args):
    m,e,l,o,q=args;target=torch.empty_like(o.data)
    for b in range(m.shape[0]):
        k=(int(l.data[b])+q-1)//q
        weight=torch.softmax(e.data[b,:,:k].double(),dim=1)
        target[b]=(weight[:,:,None]*m.data[b,:,:k,:].double()).sum(dim=1).float()
    return target


def checks(task="flash_decode2_phi"):
    source=HERE.parent/task/(task+"_repaired.py")
    source_node=next(n for n in ast.parse(source.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name.startswith("_fwd_kernel"))
    hashes={"flash_decode2_phi":"643dfb229a4f769a91dc36fd0815a8ff92f6d2e2671aa5b7627b9ec4aad1f922",
        "flash_decode2_llama":"291edeb85857e096f069b1dbe4ac55ef65a9afb388825d3c20f40300d281da02"}
    assert hashlib.sha256(ast.dump(source_node,include_attributes=False).encode()).hexdigest()==hashes[task]
    call,kernel,context=harness(source);admissions=0;rejections=0
    dims=(1,7,16,33,128) if task.endswith("phi") else (16,32,64,128)
    cases=[]
    for d in dims:cases.append(sample(2,3,4,d,[1,15]))
    cases.append(sample(1,1,1,dims[-1],[2**31-1],2**31-1))
    cases.append(sample(2,1,4,dims[0],[5,16]))
    args,backing=sample(2,2,4,dims[0],[1,16])
    args[0].data[0,:,1:,:]=float("nan");args[1].data[0,:,1:]=float("nan")
    cases.append((args,backing))
    # Contiguous singleton axes may have noncanonical unused strides.
    args,backing=sample(1,1,3,dims[0],[10])
    for i in (0,1,3):
        t=args[i].data;strides=list(t.stride())
        for axis,n in enumerate(t.shape):
            if n==1:strides[axis]=9001+axis
        args[i]=Tensor(t.as_strided(t.shape,strides))
    cases.append((args,backing))
    # Read-only summaries may share one backing allocation.
    args,backing=sample(1,1,3,dims[0],[10]);args[1]=Tensor(args[0].data.reshape(-1)[:3].view(1,1,3))
    cases.append((args,backing))
    for args,backing in cases:
        answer=expected(args);saved=[bytes(t.data.untyped_storage()) for t in args[:3]]
        original=args[3];result=call(*args)
        assert result is None and args[3] is original and context.current==7
        torch.testing.assert_close(args[3].data,answer,rtol=3e-5,atol=3e-6)
        assert torch.equal(backing[-13:],torch.full((13,),12345.))
        assert [bytes(t.data.untyped_storage()) for t in args[:3]]==saved
        assert args[2].content_reads==args[0].shape[0]
        admissions+=1
    for B,H in ((0,3),(2,0)):
        args,backing=sample(B,H,0,dims[0],[-1]*B)
        args[0].data=args[0].data.as_strided(args[0].shape,(100,100,100,100))
        prior=len(kernel.events);prior_context=len(context.events);saved=[bytes(t.data.untyped_storage()) for t in args[:4]]
        assert call(*args) is None and len(kernel.events)==prior and len(context.events)==prior_context
        assert [bytes(t.data.untyped_storage()) for t in args[:4]]==saved
        assert not any(t.pointer_reads or t.content_reads for t in args[:4]) and context.current==7
        admissions+=1
    controls=[]
    def control(edit):
        a,_=sample(2,2,3,dims[0],[8,9]);edit(a);controls.append(a)
    control(lambda a:a.__setitem__(0,None))
    control(lambda a:a.__setitem__(0,Tensor(a[0].data,cuda=False)))
    control(lambda a:a.__setitem__(1,Tensor(a[1].data,device=1)))
    control(lambda a:a.__setitem__(0,Tensor(a[0].data,layout=object())))
    control(lambda a:a.__setitem__(0,Tensor(a[0].data.half())))
    control(lambda a:a.__setitem__(2,Tensor(a[2].data.long())))
    control(lambda a:a.__setitem__(3,Tensor(a[3].data.half())))
    control(lambda a:a.__setitem__(0,Tensor(a[0].data.transpose(0,1))))
    control(lambda a:a.__setitem__(0,Tensor(torch.cat((torch.zeros(1),a[0].data.flatten()))[1:].view(a[0].shape))))
    control(lambda a:a.__setitem__(0,Tensor(torch._neg_view(a[0].data))))
    control(lambda a:a.__setitem__(0,Tensor(a[0].data,conjugate=True)))
    control(lambda a:a.__setitem__(0,Tensor(a[0].data.reshape(2,2,-1))))
    control(lambda a:a.__setitem__(1,Tensor(torch.ones(2,2,2))))
    control(lambda a:a.__setitem__(2,Tensor(torch.ones(1,dtype=torch.int32))))
    control(lambda a:a.__setitem__(3,Tensor(torch.ones(1,2,dims[0]))))
    for q in (False,1.0,0,-1,2**31):control(lambda a,q=q:a.__setitem__(4,q))
    for lengths in ([0,4],[-1,4],[13,4],[-2**31,4]):
        control(lambda a,lengths=lengths:a.__setitem__(2,Tensor(torch.tensor(lengths,dtype=torch.int32))))
    for pointer in (4097,(1<<64)-4):control(lambda a,pointer=pointer:a.__setitem__(0,Tensor(a[0].data,pointer=pointer)))
    control(lambda a:a.__setitem__(3,Tensor(a[0].data.flatten()[:a[3].numel()].view(a[3].shape))))
    control(lambda a:a.__setitem__(2,Tensor(a[0].data.flatten().view(torch.int32)[:2])))
    # Distinct imported storage objects can overlap at different base addresses.
    def imported_overlap(a):
        buffer=memoryview(a[0].data.numpy()).cast("B")[4:]
        a[3]=Tensor(torch.frombuffer(buffer,dtype=torch.float32)[:a[3].numel()].view(a[3].shape))
    control(imported_overlap)
    if task.endswith("llama"):controls.append(sample(1,1,1,7,[1])[0])
    else:controls.append(sample(1,1,1,0,[1])[0])
    for args in controls:
        previous=len(kernel.events)
        saved=[bytes(t.data.untyped_storage()) if isinstance(t,Tensor) else None for t in args[:4]]
        try:call(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError("accepted rejection control")
        assert len(kernel.events)==previous and context.current==7
        assert [bytes(t.data.untyped_storage()) if isinstance(t,Tensor) else None for t in args[:4]]==saved
        rejections+=1
    assert ".Proofs." not in (source.parent/"Kernels.lean").read_text()
    assert "FPEquiv" not in (source.parent/"Correct.lean").read_text().split("namespace")[0]
    assert "Correct" not in (source.parent/"FPEquiv.lean").read_text().split("namespace")[0]
    print(f"{task}: {admissions} source/summary/tail/empty/alias cases and {rejections} rejections passed; storage frames preserved")


if __name__=="__main__":checks()
