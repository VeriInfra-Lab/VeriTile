"""CPU physical-storage replay of actual candidate kernel ASTs; no native GPU claim."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'fused_recurrent_retention_repaired.py'
HIST=Path('/root/.codex/visualizations/2026/10/16/01a14539-f72c-7c83-aa71-3a0014d6e60a/veritile-work/bench/repairs/tritonbench_g/fused_recurrent_retention/fused_recurrent_retention_repaired.py')

def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

class Tensor:
    allocated=[]
    def __init__(self,data,device='cuda:0'):self.data,self.device=data,torch.device(device)
    @property
    def shape(self):return self.data.shape
    @property
    def ndim(self):return self.data.ndim
    @property
    def dtype(self):return self.data.dtype
    @property
    def layout(self):return self.data.layout
    @property
    def requires_grad(self):return self.data.requires_grad
    def is_contiguous(self):return self.data.is_contiguous()
    def is_neg(self):return self.data.is_neg()
    def is_conj(self):return self.data.is_conj()
    def data_ptr(self):return self.data.data_ptr()
    def numel(self):return self.data.numel()
    def stride(self,dim=None):return self.data.stride() if dim is None else self.data.stride(dim)
    def storage_offset(self):return self.data.storage_offset()
    def untyped_storage(self):return self.data.untyped_storage()
    def detach(self):return Tensor(self.data.detach(),str(self.device))
    def new_empty(self,*shape):return allocate(shape,self.dtype,str(self.device))
    def sum(self,dim):
        value=self.data.sum(dim)
        result=allocate(value.shape,self.dtype,str(self.device));result.data.copy_(value)
        return result

def allocate(shape,dtype=torch.float32,device='cuda:0'):
    count=math.prod(shape)
    backing=torch.full((count+8,),-444.5,dtype=dtype)
    result=Tensor(backing[4:4+count].reshape(shape),device)
    result.data.fill_(float('nan'))
    Tensor.allocated.append((result,backing))
    return result

def physical(tensor):
    return torch.as_strided(tensor,(tensor.untyped_storage().nbytes()//4,),(1,),storage_offset=0)

class Pointer:
    def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,torch.as_tensor(offset,dtype=torch.int64)
    @property
    def dtype(self):return SimpleNamespace(element_ty=self.tensor.dtype)
    def __add__(self,offset):
        result=Pointer(self.tensor,self.offset+offset)
        assert bool((result.offset>=-2**61).all()) and bool((result.offset<2**61).all())
        addresses=self.tensor.data_ptr()+4*result.offset
        assert bool((addresses>=0).all()) and bool((addresses<=2**63-1).all())
        return result
    def __sub__(self,offset):return self+-offset

class Context:
    def save_for_backward(self,*tensors):self.saved_tensors=tensors

def checks():
    spec=importlib.util.spec_from_file_location('retention_candidate',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=('fused_recurrent_retention_fwd_kernel','fused_recurrent_retention_bwd_kernel')
    for name in names:
        if HIST.exists():assert ast.dump(function(SOURCE,name))==ast.dump(function(HIST,name))
    for name in ('Kernels.lean','AddressSource.lean','AddressCheck.lean','APISource.lean','Contract.lean'):
        assert not any('.Proofs.' in line for line in (HERE/name).read_text().splitlines() if line.startswith('import '))
    pid=[(0,0,0)];programs=[0];events=[];current_device=[0];context_entries=[]
    class DeviceContext:
        def __init__(self,device):self.device=torch.device(device)
        def __enter__(self):
            self.previous=current_device[0];current_device[0]=self.device.index
            context_entries.append(self.device.index)
        def __exit__(self,*exc):current_device[0]=self.previous
    def load(ptr,mask=None,other=0):
        mask=torch.ones_like(ptr.offset,dtype=torch.bool) if mask is None else mask
        off=ptr.offset[mask]+ptr.tensor.storage_offset();backing=physical(ptr.tensor)
        assert bool((off>=0).all()) and bool((off<backing.numel()).all())
        result=torch.full(ptr.offset.shape,other,dtype=ptr.tensor.dtype)
        result[mask]=backing[off];events.append('load');return result
    def store(ptr,value,mask=None):
        mask=torch.ones_like(ptr.offset,dtype=torch.bool) if mask is None else mask
        off=ptr.offset[mask]+ptr.tensor.storage_offset();backing=physical(ptr.tensor)
        assert bool((off>=0).all()) and bool((off<backing.numel()).all())
        assert len(set(off.flatten().tolist()))==off.numel()
        backing[off]=value.expand(ptr.offset.shape)[mask];events.append('store')
    namespace={'tl':SimpleNamespace(constexpr=object,float32=torch.float32,
        math=SimpleNamespace(exp2=lambda x:torch.exp2(torch.as_tensor(x,dtype=torch.float32))),
        program_id=lambda axis:pid[0][axis],arange=lambda lo,hi:torch.arange(lo,hi,dtype=torch.int64),
        zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),load=load,store=store,
        sum=lambda x,axis=None:x.sum(dim=axis) if axis is not None else x.sum(),debug_barrier=lambda:events.append('barrier'))}
    for name in names:
        node=function(SOURCE,name);node.decorator_list=[]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
        class Launch:
            def __init__(self,name):self.name=name
            def __getitem__(self,grid):
                def run(*args,**kwargs):
                    assert current_device[0]==args[0].device.index
                    kwargs.pop('num_warps');kwargs.pop('num_stages')
                    for bh in reversed(range(grid[2])):
                        for ik in reversed(range(grid[1])):
                            for iv in reversed(range(grid[0])):
                                pid[0]=(iv,ik,bh);programs[0]+=1
                                namespace[self.name](*[Pointer(x.data) if isinstance(x,Tensor) else x for x in args],**kwargs)
                return run
        setattr(module,name,Launch(name))
    module.torch=SimpleNamespace(Tensor=Tensor,float32=torch.float32,strided=torch.strided,is_grad_enabled=lambda:False,
        cuda=SimpleNamespace(device=DeviceContext,current_device=lambda:current_device[0]))
    contexts=[];original=module.FusedRecurrentRetentionFunction
    class Apply:
        @staticmethod
        def apply(*args):
            ctx=Context();contexts.append(ctx);return original.forward(ctx,*args)
    module.FusedRecurrentRetentionFunction=Apply
    admitted=0
    cases=[(1,1,1,1,1,False,False,'dense',False),
        (2,2,3,4,8,True,True,'strided',False),
        (1,2,2,33,65,True,True,'expanded',False),
        (2,1,2,65,33,False,True,'final_only',False),
        (1,1,4,2,4,True,False,'dense',False),
        (2,2,2,4,4,True,True,'strided',True)]
    for B,H,T,K,V,seed,final,mode,alias in cases:
        torch.manual_seed(K+V);Tensor.allocated.clear()
        q=Tensor(torch.randn(B,H,T,K)*0.2)
        k=q if alias else Tensor(torch.randn(B,H,T,K)*0.2)
        v=q if alias else Tensor(torch.randn(B,H,T,V)*0.2)
        h0=Tensor((torch.randn(B,H,K,V)*0.1).requires_grad_()) if seed else None
        device='cuda:1' if mode=='expanded' else 'cuda:0'
        for tensor in [q,k,v]+([h0] if h0 else []):tensor.device=torch.device(device)
        snapshots=[(x,physical(x.data).detach().clone()) for x in [q,k,v]+([h0] if seed else [])]
        output,ht=module.fused_recurrent_retention(q,k,v,h0,final);ctx=contexts[-1]
        if seed:
            assert ctx.saved_tensors[3] is not h0
            assert ctx.saved_tensors[3].data_ptr()==h0.data_ptr() and not ctx.saved_tensors[3].requires_grad
        if mode=='strided':
            # Multiply before slicing so the view remains strided.
            do=Tensor((torch.randn(B,H,T,V*2)*0.2)[...,::2])
            dht=Tensor((torch.randn(B,H,K*2,V*3)*0.2)[:,:,::2,::3]) if final else None
        elif mode=='expanded':
            do=Tensor(torch.tensor(0.2).expand(B,H,T,V))
            dht=Tensor(torch.tensor(0.3).expand(B,H,K,V))
        else:
            do=Tensor(torch.zeros_like(output.data) if mode=='final_only' else torch.randn_like(output.data)*0.2)
            dht=Tensor(torch.randn_like(ht.data)*0.2) if final else None
        for tensor in [do]+([dht] if dht else []):tensor.device=torch.device(device)
        snapshots += [(x,physical(x.data).clone()) for x in [do]+([dht] if dht else [])]
        actual=original.backward(ctx,do,dht)
        assert len(actual)==5 and actual[3] is None and actual[4] is None
        # Independent dense differentiable recurrence; H0 is deliberately detached.
        qr=q.data.detach().double().requires_grad_();kr=k.data.detach().double().requires_grad_();vr=v.data.detach().double().requires_grad_()
        state=torch.zeros(B,H,K,V,dtype=torch.float64) if h0 is None else h0.data.detach().double()
        decay=(1-torch.pow(2.,-5-torch.arange(H,dtype=torch.float64)))[None,:,None,None]
        outs=[]
        for t in range(T):
            state=decay*state+kr[:,:,t,:,None]*vr[:,:,t,None,:]
            outs.append(K**-0.5*torch.einsum('bhkv,bhk->bhv',state,qr[:,:,t]))
        target=torch.stack(outs,dim=2);loss=(target*do.data.double()).sum()
        if final:loss=loss+(state*dht.data.double()).sum()
        loss.backward()
        torch.testing.assert_close(output.data.double(),target,rtol=4e-5,atol=3e-7)
        if final:torch.testing.assert_close(ht.data.double(),state,rtol=4e-5,atol=3e-7)
        for got,ref in zip(actual[:3],(qr.grad,kr.grad,vr.grad)):
            torch.testing.assert_close(got.data.double(),ref,rtol=5e-5,atol=4e-7)
        if mode=='final_only':assert actual[1].data.abs().sum()>0 and actual[2].data.abs().sum()>0
        for tensor,before in snapshots:assert torch.equal(physical(tensor.data).detach(),before)
        for _,backing in Tensor.allocated:
            assert bool((backing[:4]==-444.5).all()) and bool((backing[-4:]==-444.5).all())
        again=original.backward(ctx,do,dht)
        for x,y in zip(actual[:3],again[:3]):assert torch.equal(x.data,y.data)
        assert current_device[0]==0
        admitted+=1
    rejected=0
    def make():return [Tensor(torch.ones(1,1,2,4)),Tensor(torch.ones(1,1,2,4)),Tensor(torch.ones(1,1,2,8)),None,False]
    def reject(edit):
        nonlocal rejected
        args=make();edit(args);count=programs[0]
        try:module.fused_recurrent_retention(*args)
        except (TypeError,ValueError,RuntimeError):rejected+=1
        else:raise AssertionError('expected rejection')
        assert programs[0]==count
    reject(lambda a:a.__setitem__(0,object()))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(2,4))))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(1,1,0,4))))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(1,1,2,3))))
    reject(lambda a:a.__setitem__(1,Tensor(torch.ones(1,1,2,8))))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(1,1,2,4,dtype=torch.float64))))
    reject(lambda a:setattr(a[1],'device',torch.device('cpu')))
    reject(lambda a:setattr(a[1],'device',torch.device('cuda:1')))
    reject(lambda a:a.__setitem__(1,Tensor(torch.ones(1,1,2,8)[...,::2])))
    reject(lambda a:a.__setitem__(1,Tensor(torch.ones(1,1,2,4)._neg_view())))
    reject(lambda a:a.__setitem__(1,Tensor(torch.as_strided(torch.ones(8),(1,1,2,4),(8,100,4,1)))))
    reject(lambda a:a.__setitem__(3,Tensor(torch.ones(1,1,4,4))))
    reject(lambda a:a.__setitem__(4,1))
    reject(lambda a:setattr(a[0],'data_ptr',lambda:3))
    reject(lambda a:setattr(a[0],'data_ptr',lambda:2**64-4))
    reject(lambda a:setattr(a[0],'untyped_storage',lambda:SimpleNamespace(nbytes=lambda:4)))
    for mode in ('dtype','shape','terminal','higher','pointer'):
        args=make();module.fused_recurrent_retention(*args);ctx=contexts[-1]
        do=Tensor(torch.ones(1,1,2,8));dht=None
        if mode=='dtype':do=Tensor(torch.ones(1,1,2,8,dtype=torch.float64))
        if mode=='shape':do=Tensor(torch.ones(1,1,2,4))
        if mode=='terminal':dht=Tensor(torch.ones(1,1,4,8))
        if mode=='higher':module.torch.is_grad_enabled=lambda:True
        if mode=='pointer':do.data_ptr=lambda:2**64-4
        count=programs[0]
        try:original.backward(ctx,do,dht)
        except (TypeError,ValueError,RuntimeError):rejected+=1
        else:raise AssertionError('expected backward rejection')
        assert programs[0]==count;module.torch.is_grad_enabled=lambda:False
    # New allocator handles are checked before either source submission.
    previous_new_empty=Tensor.new_empty
    def invalid_allocation(self,*shape):
        result=previous_new_empty(self,*shape)
        result.data_ptr=lambda:2**64-4
        return result
    for backward in (False,True):
        args=make()
        if backward:
            module.fused_recurrent_retention(*args);ctx=contexts[-1]
        before=[physical(x.data).clone() for x in args[:3]]
        Tensor.new_empty=invalid_allocation;count=programs[0]
        try:
            if backward:original.backward(ctx,Tensor(torch.ones(1,1,2,8)),None)
            else:module.fused_recurrent_retention(*args)
        except ValueError:rejected+=1
        else:raise AssertionError('expected allocated-pointer rejection')
        finally:Tensor.new_empty=previous_new_empty
        assert programs[0]==count
        for x,snapshot in zip(args[:3],before):assert torch.equal(physical(x.data),snapshot)
    views=((0,100,(8,8,4,1),2**40),)*6
    for shape in ((1,1,2,2**30,8),(1,65536,2,4,8),(1,1,2**29,4,8)):
        try:module._address_radius(shape,views,True,True,True)
        except ValueError:rejected+=1
        else:raise AssertionError('expected symbolic source bound rejection')
    huge_views=((2**63,2**64,(8,8,4,1),2**40),)*6
    try:module._address_radius((1,1,2,4,8),huge_views,True,True,True)
    except ValueError:rejected+=1
    else:raise AssertionError('expected formed-pointer signed64 rejection')
    assert 1 in context_entries and current_device[0]==0
    print(f'retention contract checks: {rejected} rejects, {admitted} complete admissions plus repeated backwards, {programs[0]} source programs')

if __name__=='__main__':checks()
