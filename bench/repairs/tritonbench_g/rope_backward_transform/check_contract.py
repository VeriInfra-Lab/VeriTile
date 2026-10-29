"""Checked backward RoPE, original view behavior, and actual JIT CPU replay."""
import ast
import contextlib
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'rope_backward_transform_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rope_backward_transform/rope_backward_transform.py'
spec=importlib.util.spec_from_file_location('rope_forward_fixtures',HERE.parent/'rope_transform/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
Pointer,storage,function=f.Pointer,f.storage,f.function

class Tensor(f.Tensor):
    def wrap(self,value): return Tensor(value,self.device,self.is_cuda)
    def stride(self,axis=None): return self.data.stride() if axis is None else self.data.stride(axis)
    def data_ptr(self): return self.data.data_ptr()


def host(path=SOURCE):
    kernel,ids=f.f.cpu_kernel(path,'_triton_rope')
    events=[]
    @contextlib.contextmanager
    def device(d): events.append(('device',d));yield
    class Launch:
        def __getitem__(self,grid):
            def run(*args,**kwargs):
                events.append(('launch',grid))
                arguments=[Pointer(x.data) if isinstance(x,Tensor) else x for x in args]
                for row in reversed(range(grid[0])):
                    ids[0]=row;kernel(*arguments,**kwargs)
            return run
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        cuda=SimpleNamespace(device=device)),
        'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),'_triton_rope':Launch()}
    names=('_span','_check_storage','_separate','rope_backward') if path==SOURCE else ('rope_backward',)
    nodes=[function(path,name) for name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    return ns['rope_backward'],events


def reference(x,c,s):
    half=x.shape[-1]//2
    a,b=x[...,:half],x[...,half:2*half]
    return torch.cat((a*c+b*s,b*c-a*s,x[...,2*half:]),-1)


def check_source():
    class RemoveStride(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,ast.Mult) and isinstance(node.right,ast.Name) and node.right.id in ('cos_col_stride','sin_col_stride'):
                return node.left
            return node
    node=copy.deepcopy(function(SOURCE,'_triton_rope'))
    node.args.args=[a for a in node.args.args if a.arg not in ('cos_col_stride','sin_col_stride')]
    node=RemoveStride().visit(node)
    assert ast.dump(node)==ast.dump(function(ORIGINAL,'_triton_rope'))
    body=function(SOURCE,'rope_backward')
    calls=[ast.unparse(n.func) for n in ast.walk(body) if isinstance(n,ast.Call)]
    assert not any(name.endswith(('.clone','.copy_','.resolve_neg','.to')) for name in calls)
    preparation=[ast.unparse(n) for n in ast.walk(body) if isinstance(n,ast.Call) and ast.unparse(n.func).endswith('.contiguous')]
    assert preparation==['dq.transpose(1, 2).contiguous()','dk.transpose(1, 2).contiguous()']


def check_host():
    backward,events=host()
    torch.manual_seed(79)
    cases=[(2,3,2,4,6),(2,2,3,1,6),(2,1,3,4,5),(1,3,1,5,9),
           (1,0,2,3,6),(1,2,0,3,6),(2,0,0,3,6),(0,2,3,4,6),
           (2,2,3,0,6),(2,2,3,4,1),(2,2,3,4,0),(1,3,5,3,34)]
    for index,(batch,hq,hk,seq,dim) in enumerate(cases):
        inputs=[];backing=[];before=[]
        for shape in ((batch,hq,seq,dim),(batch,hk,seq,dim)):
            count=torch.empty(shape).numel();raw=torch.randn(count+11)
            inputs.append(Tensor(raw[3:3+count].reshape(shape)));backing.append(raw);before.append(raw.clone())
        q,k=inputs; q0,k0=q.data.clone(),k.data.clone()
        half=dim//2
        cr=torch.randn(seq+2,half*2+3);sr=torch.randn(seq+3,half*3+4)
        c=Tensor(cr[:,:2*half:2]);s=Tensor(sr[:,1:1+3*half:3])
        if index==0:s=c
        cb,sb=storage(c.data).clone(),storage(s.data).clone()
        events.clear();oq,ok=backward(q,k,c,s)
        for input_,output,initial,raw,old in zip(inputs,(oq,ok),(q0,k0),backing,before):
            torch.testing.assert_close(output.data,reference(initial,c.data[:seq],s.data[:seq]),rtol=0,atol=0)
            reuse=input_.numel()==0 or seq<=1 or input_.shape[1]<=1
            assert (output.untyped_storage().data_ptr()==input_.untyped_storage().data_ptr())==reuse
            if not reuse:assert torch.equal(raw,old)
            assert torch.equal(raw[:3],old[:3]) and torch.equal(raw[3+input_.numel():],old[3+input_.numel():])
        assert torch.equal(storage(c.data),cb) and torch.equal(storage(s.data),sb)
        assert sum(x[0]=='launch' for x in events)==bool(batch*seq and dim>=2 and (hq or hk))
        assert any(x[0]=='device' for x in events)
    base=[Tensor(torch.randn(2,2,3,6)),Tensor(torch.randn(2,3,3,6)),Tensor(torch.randn(3,3)),Tensor(torch.randn(3,3))]
    failures=[]
    def bad(i,value,error):
        args=base.copy();args[i]=value;failures.append((args,error))
    bad(0,[],TypeError);bad(0,Tensor(torch.randn(2,2,3,6,dtype=torch.float64)),TypeError)
    bad(0,Tensor(torch.randn(2,2,3,12)[...,::2]),ValueError)
    bad(0,Tensor(torch._neg_view(base[0].data)),ValueError)
    bad(0,Tensor(base[0].data,'cpu',False),ValueError);bad(1,Tensor(base[1].data,'cuda:0'),ValueError)
    bad(0,Tensor(torch.randn(2,2,3)),ValueError);bad(1,Tensor(torch.randn(1,3,3,6)),ValueError)
    bad(2,Tensor(torch.randn(2,3)),ValueError);bad(3,Tensor(torch.randn(3,4)),ValueError)
    shared=Tensor(torch.randn(2,2,3,6)); failures.append(([shared,shared,base[2],base[3]],ValueError))
    aliased=Tensor(base[0].data.flatten()[:9].view(3,3));bad(2,aliased,ValueError)
    for args,error in failures:
        events.clear()
        try:backward(*args)
        except error:pass
        else:raise AssertionError('unsupported input admitted')
        assert not any(x[0]=='launch' for x in events)
    original,_=host(ORIGINAL)
    q=Tensor(torch.arange(24,dtype=torch.float32).reshape(1,2,2,6));k=Tensor(q.data.clone())
    c=Tensor(torch.arange(12,dtype=torch.float32).reshape(2,6)[:,::2]);s=Tensor(torch.ones(2,6)[:,::2])
    expected=reference(q.data.clone(),c.data,s.data)
    output,_=original(q,k,c,s)
    assert not torch.equal(output.data,expected)
    print(f'Only coefficient-stride JIT repair; {len(cases)} copy/reuse/odd-tail/empty paths; {len(failures)} rejections passed.')
    print('Original strided-coefficient defect reproduced. CPU storage replay only; CUDA mocked.')

if __name__=='__main__':check_source();check_host()
