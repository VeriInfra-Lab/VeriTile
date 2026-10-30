"""CPU storage replay of both unchanged repaired kernel ASTs; no GPU claim."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'fused_recurrent_delta_repaired.py'
HIST=Path('/root/.codex/visualizations/2026/10/16/01a14539-f72c-7c83-aa71-3a0014d6e60a/veritile-work/bench/repairs/tritonbench_g/fused_recurrent_delta/fused_recurrent_delta_repaired.py')

def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

class Tensor:
    allocated=[]
    def __init__(self,data,device='cuda:0'):
        self.data,self.device=data,torch.device(device)
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
    def __getitem__(self,key):return Tensor(self.data[key],str(self.device))
    def new_empty(self,*shape,dtype=None):return allocate(shape,dtype or self.dtype,str(self.device),False)
    def new_zeros(self,*shape,dtype=None):return allocate(shape,dtype or self.dtype,str(self.device),True)
    def squeeze(self,dim):return Tensor(self.data.squeeze(dim),str(self.device))
    def sum(self,dim):return Tensor(self.data.sum(dim),str(self.device))
    def to(self,other):
        assert self.dtype==other.dtype and self.device==other.device
        return self


def allocate(shape,dtype=torch.float32,device='cuda:0',zero=False):
    count=math.prod(shape)
    backing=torch.full((count+8,),-444.5,dtype=dtype)
    view=Tensor(backing[4:4+count].reshape(shape),device)
    view.data.fill_(0 if zero else float('nan'))
    Tensor.allocated.append((view,backing))
    return view

class Pointer:
    def __init__(self,tensor,offset=0):
        self.tensor,self.offset=tensor,torch.as_tensor(offset,dtype=torch.int64)
    @property
    def dtype(self):return SimpleNamespace(element_ty=self.tensor.dtype)
    def __add__(self,offset):
        result=Pointer(self.tensor,self.offset+offset)
        assert bool((result.offset>=-2**31).all()) and bool((result.offset<2**31).all())
        addresses=self.tensor.data_ptr()+4*result.offset
        assert bool((addresses>=0).all()) and bool((addresses<=2**63-1).all())
        return result
    def __sub__(self,offset):return self+-offset

class Context:
    def save_for_backward(self,*tensors):self.saved_tensors=tensors


def checks():
    spec=importlib.util.spec_from_file_location('delta_candidate',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('fused_recurrent_fwd_kernel','fused_recurrent_bwd_kernel'):
        if HIST.exists():assert ast.dump(function(SOURCE,name))==ast.dump(function(HIST,name))
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    pid=[(0,0,0)];programs=[0];events=[];current_device=[0];launch_devices=[]
    class DeviceContext:
        def __init__(self,device):self.device=torch.device(device)
        def __enter__(self):
            self.previous=current_device[0];current_device[0]=self.device.index
        def __exit__(self,*exc):current_device[0]=self.previous
    def load(ptr,mask=None,other=0):
        mask=torch.ones_like(ptr.offset,dtype=torch.bool) if mask is None else mask
        off=ptr.offset[mask]
        assert bool((off>=0).all()) and bool((off<ptr.tensor.numel()).all())
        result=torch.full(ptr.offset.shape,other,dtype=ptr.tensor.dtype)
        result[mask]=ptr.tensor.reshape(-1)[off]
        events.append('load');return result
    def store(ptr,value,mask=None):
        mask=torch.ones_like(ptr.offset,dtype=torch.bool) if mask is None else mask
        off=ptr.offset[mask]
        assert bool((off>=0).all()) and bool((off<ptr.tensor.numel()).all())
        assert len(set(off.flatten().tolist()))==off.numel()
        ptr.tensor.reshape(-1)[off]=value.expand(ptr.offset.shape)[mask]
        events.append('store')
    namespace={'tl':SimpleNamespace(constexpr=object,float32=torch.float32,
        program_id=lambda axis:pid[0][axis],arange=lambda lo,hi:torch.arange(lo,hi,dtype=torch.int64),
        zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),load=load,store=store,
        sum=lambda x,axis=None:x.sum(dim=axis) if axis is not None else x.sum(),
        debug_barrier=lambda:events.append('barrier'))}
    for name in ('fused_recurrent_fwd_kernel','fused_recurrent_bwd_kernel'):
        node=function(SOURCE,name);node.decorator_list=[]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
        class Launch:
            def __init__(self,name):self.name=name
            def __getitem__(self,grid):
                def run(*args,**kwargs):
                    assert current_device[0]==args[0].device.index
                    launch_devices.append((self.name,current_device[0]))
                    kwargs.pop('num_warps');kwargs.pop('num_stages')
                    for bh in reversed(range(grid[2])):
                        for ik in reversed(range(grid[1])):
                            for iv in reversed(range(grid[0])):
                                pid[0]=(iv,ik,bh);programs[0]+=1
                                namespace[self.name](*[Pointer(x.data) if isinstance(x,Tensor) else x for x in args],**kwargs)
                return run
        setattr(module,name,Launch(name))
    module.torch=SimpleNamespace(Tensor=Tensor,float32=torch.float32,strided=torch.strided,
        cuda=SimpleNamespace(device=DeviceContext),is_grad_enabled=lambda:False,ones_like=lambda x:Tensor(torch.ones_like(x.data)),
        empty_like=lambda x,dtype:allocate(x.shape,dtype,str(x.device)))
    contexts=[]
    class Apply:
        @staticmethod
        def apply(*args):
            ctx=Context();contexts.append(ctx)
            return original.forward(ctx,*args)
    original=module.FusedRecurrentFunction
    module.FusedRecurrentFunction=Apply
    admitted=0
    for B,H,T,K,V,kind,initial,final in [(1,1,1,1,1,'default',False,False),
        (1,2,3,5,7,'scalar',True,True),(1,1,3,3,33,'vector',True,True),
        (2,2,2,7,65,'vector',False,True),(1,1,2,4,9,'scalar',True,False),
        (1,1,1,1,34,'vector',True,True),(1,1,2,2,3,'scalar','detached',False)]:
        torch.manual_seed(K+V);Tensor.allocated.clear()
        q,k=[Tensor(torch.randn(B,H,T,K)*0.2) for _ in range(2)]
        v=Tensor(torch.randn(B,H,T,V)*0.2)
        beta=None if kind=='default' else Tensor(torch.rand((B,H,T,V) if kind=='vector' else (B,H,T))*0.5)
        h0=Tensor((torch.randn(B,H,K,V)*0.1).requires_grad_(initial is True)) if initial else None
        device='cuda:1' if (K,V)==(3,33) else 'cuda:0'
        for tensor in [q,k,v]+([beta] if beta else [])+([h0] if h0 else []):tensor.device=torch.device(device)
        pristine=v.data.clone();qold=q.data.clone();kold=k.data.clone()
        scale=None if kind=='default' else 0.75
        output,ht=module.fused_recurrent_delta_rule(q,k,v,beta,scale,h0,final)
        ctx=contexts[-1]
        do=Tensor(torch.randn_like(output.data)*0.2)
        dht=Tensor(torch.randn_like(ht.data)*0.2) if final else None
        do.device=q.device
        if dht is not None:dht.device=q.device
        actual=original.backward(ctx,do,dht)
        assert len(actual)==7 and actual[4] is None and actual[6] is None
        assert (actual[5] is not None)==(initial is True)
        # Independent dense differentiable recurrence; source v mutation checked separately.
        qr=q.data.double().requires_grad_();kr=k.data.double().requires_grad_();vr=pristine.double().requires_grad_()
        br=(torch.ones(B,H,T,dtype=torch.float64) if beta is None else beta.data.double()).requires_grad_()
        hr=(torch.zeros(B,H,K,V,dtype=torch.float64) if h0 is None else h0.data.detach().double()).requires_grad_()
        state=hr;outs=[];residuals=[];sc=K**-0.5 if scale is None else scale
        for t in range(T):
            residual=vr[:,:,t]-torch.einsum('bhkv,bhk->bhv',state,kr[:,:,t])
            residuals.append(residual)
            b=br[:,:,t] if kind=='vector' else br[:,:,t,None]
            state=state+kr[:,:,t,:,None]*(residual*b)[:,:,None,:]
            outs.append(sc*torch.einsum('bhkv,bhk->bhv',state,qr[:,:,t]))
        target=torch.stack(outs,dim=2)
        loss=(target*do.data.double()).sum()
        if final:loss=loss+(state*dht.data.double()).sum()
        loss.backward()
        torch.testing.assert_close(output.data.double(),target,rtol=3e-5,atol=2e-7)
        torch.testing.assert_close(v.data.double(),torch.stack(residuals,dim=2),rtol=3e-5,atol=2e-7)
        if final:torch.testing.assert_close(ht.data.double(),state,rtol=3e-5,atol=2e-7)
        for got,ref in zip(actual[:4],(qr.grad,kr.grad,vr.grad,br.grad)):
            torch.testing.assert_close(got.data.double(),ref,rtol=5e-5,atol=3e-7)
        if initial is True:torch.testing.assert_close(actual[5].data.double(),hr.grad,rtol=5e-5,atol=3e-7)
        assert torch.equal(q.data,qold) and torch.equal(k.data,kold)
        for _,backing in Tensor.allocated:
            assert bool((backing[:4]==-444.5).all()) and bool((backing[-4:]==-444.5).all())
        try:original.backward(ctx,do,dht)
        except RuntimeError:pass
        else:raise AssertionError('repeated backward admitted')
        assert current_device[0]==0
        admitted+=1
    rejected=0
    def make():
        return [Tensor(torch.ones(1,1,2,3)),Tensor(torch.ones(1,1,2,3)),Tensor(torch.ones(1,1,2,5)),None,None,None,False]
    def reject(edit):
        nonlocal rejected
        args=make();edit(args);count=programs[0]
        try:module.fused_recurrent_delta_rule(*args)
        except (TypeError,ValueError,RuntimeError):rejected+=1
        else:raise AssertionError('expected rejection')
        assert programs[0]==count
    reject(lambda a:a.__setitem__(0,object()))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(2,3))))
    reject(lambda a:a.__setitem__(2,Tensor(torch.ones(1,1,0,5))))
    reject(lambda a:a.__setitem__(1,Tensor(torch.ones(1,1,2,4))))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(1,1,2,3,dtype=torch.float64))))
    reject(lambda a:setattr(a[1],'device',torch.device('cpu')))
    reject(lambda a:setattr(a[1],'device',torch.device('cuda:1')))
    reject(lambda a:a.__setitem__(1,Tensor(torch.ones(1,1,2,6)[...,::2])))
    reject(lambda a:a.__setitem__(1,Tensor(torch.ones(1,1,2,3)._neg_view())))
    reject(lambda a:a.__setitem__(1,a[0]))
    reject(lambda a:a.__setitem__(1,Tensor(torch.as_strided(torch.ones(6),(1,1,2,3),(6,100,3,1)))))
    reject(lambda a:a.__setitem__(3,Tensor(torch.ones(1,1,2,4))))
    reject(lambda a:a.__setitem__(5,Tensor(torch.ones(1,1,3,4))))
    for value in (0.0,-1.0,float('inf'),float('nan'),1e100,True,1):
        reject(lambda a,v=value:a.__setitem__(4,v))
    reject(lambda a:a.__setitem__(6,1))
    reject(lambda a:setattr(a[0],'data_ptr',lambda:3))
    reject(lambda a:setattr(a[0],'data_ptr',lambda:2**64-4))
    # Concrete old-admission counterexample: contiguous singleton H has an
    # arbitrary stride, but the source flattens B*H using that stride.
    witness_q=Tensor(torch.as_strided(torch.ones(12),(2,1,2,3),(6,100,3,1)))
    witness_args=[witness_q,Tensor(torch.ones(2,1,2,3)),Tensor(torch.ones(2,1,2,5)),None,None,None,False]
    assert witness_q.is_contiguous() and witness_q.numel()==12 and witness_q.stride(1)==100
    assert 1*witness_q.stride(1)>=witness_q.numel()
    old_guard=function(SOURCE,'_tensor')
    old_guard.body=[node for node in old_guard.body if not (isinstance(node,ast.If) and
        isinstance(node.test,ast.Compare) and ast.unparse(node.test).startswith('tensor.stride() !='))]
    previous=module._tensor
    exec(compile(ast.fix_missing_locations(ast.Module(body=[old_guard],type_ignores=[])),str(SOURCE),'exec'),module.__dict__)
    module._forward_inputs(*witness_args)  # All pre-correction guards pass.
    try:
        module.fused_recurrent_delta_rule(*witness_args)
    except AssertionError:
        pass  # Actual source AST reaches q offsets 100..102 in a 12-word view.
    else:
        raise AssertionError('old admission failed to expose source OOB')
    module._tensor=previous
    count=programs[0]
    try:module.fused_recurrent_delta_rule(*witness_args)
    except ValueError:rejected+=1
    else:raise AssertionError('corrected singleton stride guard missing')
    assert programs[0]==count
    for mode in ('expanded', 'dtype', 'terminal', 'higher'):
        args=make(); module.fused_recurrent_delta_rule(*args);ctx=contexts[-1]
        do=Tensor(torch.ones(1,1,2,5));dht=None
        if mode=='expanded':do=Tensor(torch.ones(1).expand(1,1,2,5))
        if mode=='dtype':do=Tensor(torch.ones(1,1,2,5,dtype=torch.float64))
        if mode=='terminal':dht=Tensor(torch.ones(1,1,3,5))
        if mode=='higher':module.torch.is_grad_enabled=lambda:True
        count=programs[0]
        try:original.backward(ctx,do,dht)
        except (TypeError,ValueError,RuntimeError):rejected+=1
        else:raise AssertionError('expected backward rejection')
        assert programs[0]==count
        module.torch.is_grad_enabled=lambda:False
    for shape in ((1,1,2,2**30,1),(1,1,2,2**16,32)):
        try:module._spans(shape,True)
        except ValueError:rejected+=1
        else:raise AssertionError('expected source index rejection')
    assert all((name,1) in launch_devices for name in ('fused_recurrent_fwd_kernel','fused_recurrent_bwd_kernel'))
    assert current_device[0]==0
    print(f'delta contract checks: {rejected} rejects, {admitted} forward/backward admissions, {programs[0]} program executions')

if __name__=='__main__':checks()
