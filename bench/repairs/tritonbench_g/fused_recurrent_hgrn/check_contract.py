"""Replay complete guarded HGRN kernels on CPU storage; CUDA is mocked."""
import ast
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'fused_recurrent_hgrn_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/fused_recurrent_hgrn/fused_recurrent_hgrn.py'
spec=importlib.util.spec_from_file_location('hgrn_fixture',HERE.parent/'chunk_gate_recurrence/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class Tensor(f.Tensor):
    allocator=None
    def new_empty(self,*shape):return self.allocator(shape,self.device,self.dtype)
    def detach(self):return Tensor(self.data.detach(),self.device,self.is_cuda)


def host(path=SOURCE,bd=32,current_device=1):
    events,allocations,ids=[],[],[0,0]
    def allocate(shape,device,dtype,zero=False):
        count=math.prod(shape)
        raw=torch.full((count+7,),-811.,dtype=dtype)
        data=raw[:count].reshape(shape)
        if zero:data.zero_()
        value=Tensor(data,device)
        allocations.append((value,raw));return value
    Tensor.allocator=staticmethod(allocate)
    def load(pointer,mask,other=0):
        offset=pointer.offset
        result=torch.full_like(offset,other,dtype=pointer.tensor.dtype)
        result[mask]=f.f.storage(pointer.tensor)[pointer.tensor.storage_offset()+offset[mask]]
        return result
    def store(pointer,value,mask):
        f.f.storage(pointer.tensor)[pointer.tensor.storage_offset()+pointer.offset[mask]]=value[mask]
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=lambda t:allocate(t.shape,t.device,t.dtype),zeros_like=lambda t:allocate(t.shape,t.device,t.dtype,True),
        cuda=SimpleNamespace(current_device=lambda:current_device),autograd=SimpleNamespace(Function=torch.autograd.Function)),
        'triton':SimpleNamespace(cdiv=lambda x,y:(x+y-1)//y),
        'tl':SimpleNamespace(program_id=lambda axis:ids[axis],arange=torch.arange,zeros=torch.zeros,
            float32=torch.float32,load=load,store=store)}
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node.returns=None
            for argument in node.args.args:argument.annotation=None
            if node.name.endswith('_kernel'):node.decorator_list=[]
            return self.generic_visit(node)
    body=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and not getattr(node,'name','').startswith('test_'):
            body.append(Strip().visit(node))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),ns)
    class Launch:
        def __init__(self,kernel,name):self.kernel,self.name=kernel,name
        def __getitem__(self,grid):
            def run(*args,**kwargs):
                launch=grid({'BD':bd})
                events.append((self.name,launch))
                args=[f.Pointer(value.data) if isinstance(value,Tensor) else value for value in args]
                kwargs['BD']=bd
                for p in itertools.product(*(reversed(range(n)) for n in launch)):
                    ids[:]=p;self.kernel(*args,**kwargs)
            return run
    for name in ('fused_recurrent_hgrn_fwd_kernel','fused_recurrent_hgrn_bwd_kernel'):
        ns[name]=Launch(ns[name],name)
    def context():
        ctx=SimpleNamespace()
        ctx.save_for_backward=lambda *values:setattr(ctx,'saved_tensors',values)
        ctx.set_materialize_grads=lambda flag:setattr(ctx,'materialize',flag)
        return ctx
    return ns['FusedRecurrentHGRNFunction'],context,events,allocations


def reference(x,g,seed,do,dht,final):
    b,h,t,d=x.shape
    state=torch.zeros((b,h,d)) if seed is None else seed.clone()
    output=torch.empty_like(x);previous=[]
    for i in range(t):
        previous.append(state)
        state=g[:,:,i]*state+x[:,:,i]
        output[:,:,i]=state
    dx=torch.empty_like(x);dg=torch.empty_like(g)
    carry=dht.clone() if final and dht is not None else torch.zeros_like(state)
    for i in reversed(range(t)):
        carry=carry+(torch.zeros_like(state) if do is None else do[:,:,i])
        dx[:,:,i]=carry;dg[:,:,i]=carry*previous[i]
        carry=carry*g[:,:,i]
    return output,state if final else None,dx,dg


def main():
    torch.manual_seed(404)
    ast_kernel=lambda path,name:f.f.f.function(path,name)
    assert ast.dump(ast_kernel(SOURCE,'fused_recurrent_hgrn_fwd_kernel'))==ast.dump(ast_kernel(ORIGINAL,'fused_recurrent_hgrn_fwd_kernel'))
    assert '.contiguous()' not in SOURCE.read_text() and 'enable_fp_fusion' not in SOURCE.read_text()
    assert 'initial_state = initial_state.detach()' in SOURCE.read_text()
    paths=0
    for bd in (32,64,128):
        program,context,events,allocations=host(bd=bd)
        for b,h,t,d in ((1,1,0,1),(1,2,1,33),(2,2,3,65),(1,1,2,129)):
            for use_seed,final,has_do in itertools.product((False,True),repeat=3):
                for has_dht in ((False,True) if final else (False,)):
                    x=Tensor(torch.randint(-2,3,(b,h,t,d)).float())
                    g=Tensor(torch.randint(-1,2,(b,h,t,d)).float())
                    seed=Tensor(torch.randint(-2,3,(b,h,d)).float()) if use_seed else None
                    do=Tensor(torch.randint(-2,3,(b,h,t,d)).float()) if has_do else None
                    dht=Tensor(torch.randint(-2,3,(b,h,d)).float()) if has_dht else None
                    inputs=[x,g,*[a for a in (seed,do,dht) if a is not None]]
                    old=[f.f.storage(a.data).clone() for a in inputs]
                    expected=reference(x.data,g.data,None if seed is None else seed.data,
                        None if do is None else do.data,None if dht is None else dht.data,final)
                    ctx=context();events.clear();allocations.clear()
                    out,ht=program.forward(ctx,x,g,seed,final)
                    dx,dg,dh0,dflag=program.backward(ctx,do,dht)
                    assert dh0 is None and dflag is None and ctx.saved_tensors[1] is out
                    for actual,wanted in zip((out,ht,dx,dg),expected):
                        assert (actual is None)==(wanted is None)
                        if actual is not None:assert torch.equal(actual.data,wanted),(bd,b,h,t,d,use_seed,final,has_do,has_dht)
                    for actual,wanted in zip(inputs,old):assert torch.equal(f.f.storage(actual.data),wanted)
                    for tensor,raw in allocations:assert torch.equal(raw[tensor.numel():],torch.full((7,),-811.))
                    paths+=1
    # Readonly overlapping views and actual saved-output cotangent aliases.
    for bd in (32,64,128):
        for t,d in ((1,33),(3,65)):
            program,context,_,allocations=host(bd=bd)
            raw=torch.randint(-1,2,(t*d+9,)).float()
            x=Tensor(raw[3:3+t*d].reshape(1,1,t,d))
            g=Tensor(x.data)
            seed=Tensor(raw[3:3+d].reshape(1,1,d))
            before=raw.clone()
            ctx=context();out,ht=program.forward(ctx,x,g,seed,True)
            expected=reference(x.data,g.data,seed.data,out.data,ht.data,True)
            saved=(out.data.clone(),ht.data.clone())
            dx,dg,dh0,dflag=program.backward(ctx,out,ht)
            assert torch.equal(dx.data,expected[2]) and torch.equal(dg.data,expected[3])
            assert torch.equal(raw,before) and torch.equal(out.data,saved[0]) and torch.equal(ht.data,saved[1])
            assert ctx.saved_tensors[1] is out and dh0 is None and dflag is None
            for tensor,backing in allocations:
                assert torch.equal(backing[tensor.numel():],torch.full((7,),-811.))
            paths+=1
    program,context,events,_=host()
    x=Tensor(torch.ones(1,1,2,33));g=Tensor(torch.ones_like(x.data));seed=Tensor(torch.ones(1,1,33))
    failures=[(Tensor(x.data.double()),g,seed,False),(x,Tensor(g.data.transpose(-1,-2)),seed,False),
        (x,g,Tensor(torch._neg_view(seed.data)),False),(x,g,seed,1),
        (Tensor(x.data,'cpu',False),g,None,False)]
    for rem in (1,2,3):
        storage=torch.zeros(33*4+8,dtype=torch.uint8).untyped_storage()[rem:rem+33*4]
        value=torch.empty(0).set_(storage,0,(1,1,33),(33,33,1))
        failures.append((x,g,Tensor(value),False))
    for args in failures:
        events.clear()
        try:program.forward(context(),*args)
        except ValueError:pass
        else:raise AssertionError('unsupported input admitted')
        assert not events
    ctx=context();out,_=program.forward(ctx,x,g,seed,True)
    backward_failures=[(Tensor(torch.ones(1,1,2,66)[...,::2]),None),
        (None,Tensor(torch.ones(1,1,66)[...,::2])),(Tensor(x.data.double()),None)]
    for do,dht in backward_failures:
        events.clear()
        try:program.backward(ctx,do,dht)
        except ValueError:pass
        else:raise AssertionError('unsupported cotangent admitted')
        assert not events
    original,ctx_old,_,_=host(ORIGINAL)
    ctx=ctx_old();original.forward(ctx,x,g,seed,True)
    old_dx,_,_,_=original.backward(ctx,Tensor(torch.zeros_like(x.data)),Tensor(torch.ones_like(seed.data)))
    expected=reference(x.data,g.data,seed.data,torch.zeros_like(x.data),torch.ones_like(seed.data),True)
    assert not torch.equal(old_dx.data,expected[2]), 'original ignored-final-cotangent defect was not reproduced'
    print(f'{paths} complete CPU paths across all three BD values; {len(failures)+len(backward_failures)} prelaunch rejections.')
    print('Original ignored-final-cotangent error reproduced; detached seed, optional returns, caller frame and tails passed.')
    print('CUDA is mocked; GPU arithmetic and compiler conformance are not tested.')


if __name__=='__main__':main()
