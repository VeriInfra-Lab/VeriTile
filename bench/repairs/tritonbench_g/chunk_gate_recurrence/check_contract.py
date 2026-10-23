"""Complete original-shaped recurrence replay with CPU storage and mocked CUDA."""
import ast
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'chunk_gate_recurrence_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunk_gate_recurrence/chunk_gate_recurrence.py'
spec=importlib.util.spec_from_file_location('chunk_fixture',HERE.parent/'rotary_transform/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class Tensor(f.Tensor):
    def wrap(self,value):return self if value is self.data else Tensor(value,self.device,self.is_cuda)
    def view(self,*shape):return self.wrap(self.data.view(*shape))
    def sum(self,*args,**kwargs):return self.wrap(self.data.sum(*args,**kwargs))
    def __setitem__(self,index,value):self.data[index]=value.data if isinstance(value,Tensor) else value

class Pointer:
    def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
    @property
    def dtype(self):return SimpleNamespace(element_ty=self.tensor.dtype)
    def __add__(self,offset):return Pointer(self.tensor,self.offset+offset)
    def __sub__(self,offset):return Pointer(self.tensor,self.offset-offset)


def host(path=SOURCE,current_device=1):
    events,allocations,ids=[],[],[0,0,0]
    def allocate(shape,device,dtype,zero=False):
        count=math.prod(shape)
        raw=torch.full((count+7,),-987.,dtype=dtype)
        data=raw[:count].reshape(shape)
        if zero:data.zero_()
        tensor=Tensor(data,device)
        allocations.append((tensor,raw))
        return tensor
    def empty(*shape,device,dtype):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):shape=shape[0]
        return allocate(shape,device,dtype)
    def zeros(shape,device,dtype):return allocate(shape,device,dtype,True)
    def load(pointer):return f.storage(pointer.tensor)[pointer.tensor.storage_offset()+pointer.offset]
    def store(pointer,value):f.storage(pointer.tensor)[pointer.tensor.storage_offset()+pointer.offset]=value
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty=empty,zeros=zeros,empty_like=lambda t:allocate(t.shape,t.device,t.dtype),
        zeros_like=lambda t:allocate(t.shape,t.device,t.dtype,True),
        cuda=SimpleNamespace(current_device=lambda:current_device),autograd=SimpleNamespace(Function=torch.autograd.Function)),
        'tl':SimpleNamespace(program_id=lambda axis:ids[axis],arange=torch.arange,zeros=torch.zeros,
            float32=torch.float32,load=load,store=store,sum=torch.sum)}
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node.returns=None
            for argument in node.args.args:argument.annotation=None
            if node.name in ('_fwd_recurrence','_bwd_recurrence'):node.decorator_list=[]
            return self.generic_visit(node)
    body=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and not getattr(node,'name','').startswith('test_'):
            body.append(Strip().visit(node))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),ns)
    for name in ('_fwd_recurrence','_bwd_recurrence'):
        kernel=ns[name]
        class Launch:
            def __init__(self,kernel,name):self.kernel,self.name=kernel,name
            def __getitem__(self,grid):
                def run(*args,**kwargs):
                    events.append((self.name,grid))
                    args=[Pointer(value.data) if isinstance(value,Tensor) else value for value in args]
                    kwargs={key:Pointer(value.data) if isinstance(value,Tensor) else value for key,value in kwargs.items()}
                    for p in itertools.product(*(reversed(range(n)) for n in grid)):
                        ids[:]=p;self.kernel(*args,**kwargs)
                return run
        ns[name]=Launch(kernel,name)
    def context():
        ctx=SimpleNamespace()
        ctx.save_for_backward=lambda *values:setattr(ctx,'saved_tensors',values)
        return ctx
    return ns['ChunkGateRecurrent'],context,events,allocations


def reference(x,g,seed,incoming):
    b,h,n,k,v=x.shape
    output=torch.empty_like(x)
    carry=torch.zeros((b,h,k,v)) if seed is None else seed.clone()
    for t in range(n):
        output[:,:,t]=carry
        carry=carry*g[:,:,t,None,None]+x[:,:,t]
    dx=torch.zeros_like(x);dg=torch.zeros_like(g);ds=torch.zeros((b,h,k,v))
    for t in reversed(range(n)):
        if t+1<n:
            dx[:,:,t]=ds
            dg[:,:,t]=(ds*output[:,:,t]).sum((-1,-2))
        ds=ds*g[:,:,t,None,None]+incoming[:,:,t]
    return output,dx,dg,None if seed is None else ds


def main():
    program,context,events,allocations=host()
    def kernel_ast(path,name):return f.f.function(path,name)
    assert ast.dump(kernel_ast(SOURCE,'_bwd_recurrence'))==ast.dump(kernel_ast(ORIGINAL,'_bwd_recurrence'))
    assert 'enable_fp_fusion' not in SOURCE.read_text()
    torch.manual_seed(319)
    paths=0
    for b,h,n,k,v in [(1,1,0,64,16),(1,1,1,64,16),(1,2,2,64,32),(2,2,4,128,32)]:
        for mode in ('none','dense','transpose','expand','offset'):
            x=Tensor(torch.randint(-2,3,(b,h,n,v,k)).float().transpose(-1,-2))
            gate=Tensor(torch.randint(-1,2,(b,h,n*2)).float()[...,::2])
            incoming=Tensor(torch.randint(-2,3,(b,h,n,v,k)).float().transpose(-1,-2))
            if mode=='none':seed=None
            elif mode=='transpose':seed=Tensor(torch.randint(-2,3,(b,h,v,k)).float().transpose(-1,-2))
            elif mode=='expand':seed=Tensor(torch.randint(-2,3,(1,h,k,v)).float().expand(b,h,k,v))
            elif mode=='offset':seed=Tensor(torch.randint(-2,3,(b*h*k*v+11,)).float()[5:-6].reshape(b,h,k,v))
            else:seed=Tensor(torch.randint(-2,3,(b,h,k,v)).float())
            callers=[x,gate,incoming]+([] if seed is None else [seed])
            originals=[f.storage(t.data).clone() for t in callers]
            expected=reference(x.data,gate.data,None if seed is None else seed.data,incoming.data)
            events.clear();allocations.clear();ctx=context()
            output=program.forward(ctx,x,gate,seed)
            dx,dg,ds=program.backward(ctx,incoming)
            actual=[output,dx,dg,ds]
            for a,e in zip(actual,expected):
                assert (a is None)==(e is None)
                if a is not None:assert torch.equal(a.data,e),(b,h,n,k,v,mode)
            assert ctx.saved_tensors[0] is output
            assert ctx.saved_tensors[1].shape==gate.shape
            assert len(events)==(2 if n else 0)
            for t,before in zip(callers,originals):assert torch.equal(f.storage(t.data),before)
            for t,raw in allocations:assert torch.equal(raw[t.numel():],torch.full((7,),-987.))
            paths+=1
    # Original readonly arguments can share backing storage. Incoming gradients
    # arrive after forward and may be the returned/saved output itself.
    for incoming_alias in (False,True):
        x=Tensor(torch.randint(-2,3,(1,1,3,64,16)).float())
        gate=Tensor(x.data.reshape(-1)[:3].view(1,1,3))
        seed=Tensor(x.data[:,:,0])
        original_storage=f.storage(x.data).clone()
        events.clear();allocations.clear();ctx=context()
        output=program.forward(ctx,x,gate,seed)
        incoming=output if incoming_alias else x
        expected=reference(x.data,gate.data,seed.data,incoming.data)
        before_output=output.data.clone()
        dx,dg,ds=program.backward(ctx,incoming)
        for actual,wanted in zip((output,dx,dg,ds),expected):assert torch.equal(actual.data,wanted)
        assert torch.equal(output.data,before_output)
        assert torch.equal(f.storage(x.data),original_storage)
        paths+=1
    x=Tensor(torch.ones(1,1,2,64,16));gate=Tensor(torch.ones(1,1,2));seed=Tensor(torch.ones(1,1,64,16))
    failures=[(Tensor(x.data.double()),gate,seed),(x,Tensor(torch.ones(1,2,2)),seed),
        (x,gate,Tensor(torch._neg_view(seed.data))),(Tensor(torch.ones(1,1,2,63,16)),gate,None),
        (Tensor(x.data,'cpu',False),gate,None)]
    for remainder in (1,2,3):
        raw=torch.zeros(seed.numel()*4+8,dtype=torch.uint8).untyped_storage()[remainder:remainder+seed.numel()*4]
        value=torch.empty(0).set_(raw,0,seed.shape,seed.stride())
        failures.append((x,gate,Tensor(value)))
    for args in failures:
        events.clear()
        try:program.forward(context(),*args)
        except (ValueError,TypeError):pass
        else:raise AssertionError('unsupported input admitted')
        assert not events
    wrong,wrong_context,wrong_events,_=host(current_device=0)
    try:wrong.forward(wrong_context(),x,gate,seed)
    except ValueError:pass
    else:raise AssertionError('ambient CUDA device mismatch admitted')
    assert not wrong_events
    ctx=context();output=program.forward(ctx,x,gate,seed)
    backward_failures=(Tensor(torch.ones(1,1,2,64,16,dtype=torch.float64)),
        Tensor(torch.ones(1,1,2,63,16)))
    for incoming in backward_failures:
        events.clear();before=output.data.clone()
        try:program.backward(ctx,incoming)
        except ValueError:pass
        else:raise AssertionError('unsupported incoming gradient admitted')
        assert not events and torch.equal(output.data,before)
    ranges=program.forward.__globals__['_source_ranges']
    assert all(r.valid for r in ranges((1,1,1,64,16),(1024,1024,16,1)))
    assert not all(r.valid for r in ranges((1,1,2**31,64,16),(1024,1024,16,1)))
    assert not all(r.valid for r in ranges((1,1,2,64,16),(0,0,2**31,1)))
    valid=ranges((1,1,1,64,16),(0,0,0,0))
    assert valid[11].lo<0 and valid[13].lo<0, 'N=1 dead reverse pointers must remain checked'
    original,original_context,_,_=host(ORIGINAL)
    seed_transpose=Tensor(torch.arange(1024.).reshape(1,1,16,64).transpose(-1,-2))
    original_output=original.forward(original_context(),x,gate,seed_transpose)
    assert not torch.equal(original_output.data[:,:,0],seed_transpose.data)
    empty=Tensor(torch.empty(1,1,0,64,16));empty_gate=Tensor(torch.empty(1,1,0))
    try:original.forward(original_context(),empty,empty_gate,None)
    except IndexError:pass
    else:raise AssertionError('original N=0 invalid first store was not reproduced')
    print('Original strided-seed error and N=0 invalid first store reproduced.')
    print(f'{paths} complete forward/backward CPU paths; {len(failures)+1+len(backward_failures)} prelaunch rejections.')
    print('Saved output flow, optional seed gradient, caller frame and allocation tails passed. CUDA/numerics not tested.')


if __name__=='__main__':main()
