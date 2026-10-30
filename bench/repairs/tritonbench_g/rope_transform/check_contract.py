"""Original RoPE JIT and checked host replay using CPU physical storage."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'rope_transform_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rope_transform/rope_transform.py'
spec=importlib.util.spec_from_file_location('rope_fixtures',HERE.parent/'swiglu_fwd/check_contract.py')
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
Pointer,storage,function=f.Pointer,f.storage,f.function

class Tensor(f.Tensor):
    def wrap(self,value): return Tensor(value,self.device,self.is_cuda)
    def transpose(self,*axes): return self.wrap(self.data.transpose(*axes))
    def contiguous(self):
        value=self.data.contiguous()
        return self if value is self.data else self.wrap(value)
    def is_contiguous(self): return self.data.is_contiguous()
    def is_conj(self): return self.data.is_conj()
    def numel(self): return self.data.numel()


def host(path=SOURCE):
    kernel,ids=f.cpu_kernel(path,'_triton_rope')
    events=[]
    @contextlib.contextmanager
    def device(d): events.append(('device',d)); yield
    class Launch:
        def __getitem__(self,grid):
            def run(q,qs,k,ks,cos,cs,sin,ss,*args,**kwargs):
                events.append(('launch',grid,qs,ks))
                for row in reversed(range(grid[0])):
                    ids[0]=row
                    kernel(Pointer(q.data),qs,Pointer(k.data),ks,Pointer(cos.data),cs,
                        Pointer(sin.data),ss,*args,**kwargs)
            return run
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        cuda=SimpleNamespace(device=device)),
        'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),'_triton_rope':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function(path,'rope_forward')],type_ignores=[])),str(path),'exec'),ns)
    return ns['rope_forward'],events


def reference(x,c,s):
    half=x.shape[-1]//2
    a,b=x[...,:half],x[...,half:]
    return torch.cat((a*c-b*s,b*c+a*s),-1)


def check_host():
    assert ast.dump(function(SOURCE,'_triton_rope'))==ast.dump(function(ORIGINAL,'_triton_rope'))
    forward,events=host()
    torch.manual_seed(73)
    shapes=[(2,3,2,4,6),(2,2,3,1,2),(2,1,3,4,6),(1,3,1,5,10),
            (1,1,1,1,2),(0,2,3,4,6),(2,2,3,0,6),(1,3,5,3,34)]
    for batch,hq,hk,seq,dim in shapes:
        tensors=[]; backing=[]; before=[]
        for shape in ((batch,hq,seq,dim),(batch,hk,seq,dim),(seq,dim//2),(seq,dim//2)):
            n=torch.empty(shape).numel()
            raw=torch.randn(n+11); t=Tensor(raw[3:3+n].reshape(shape))
            tensors.append(t); backing.append(raw); before.append(raw.clone())
        q,k,c,s=tensors
        q0,k0=q.data.clone(),k.data.clone()
        events.clear(); oq,ok,oc,os=forward(q,k,c,s)
        assert oc is c and os is s
        for input_,output,initial,raw,old in zip((q,k),(oq,ok),(q0,k0),backing,before):
            torch.testing.assert_close(output.data,reference(initial,c.data,s.data),rtol=0,atol=0)
            reuse=input_.numel()==0 or seq==1 or input_.shape[1]==1
            same=output.untyped_storage().data_ptr()==input_.untyped_storage().data_ptr()
            assert same==reuse
            if not reuse: assert torch.equal(raw,old)
            assert torch.equal(raw[:3],old[:3]) and torch.equal(raw[3+input_.numel():],old[3+input_.numel():])
        assert torch.equal(backing[2],before[2]) and torch.equal(backing[3],before[3])
        assert sum(x[0]=='launch' for x in events)==bool(batch*seq)
        assert any(x[0]=='device' for x in events)
    base=[Tensor(torch.randn(2,2,3,6)),Tensor(torch.randn(2,3,3,6)),Tensor(torch.randn(3,3)),Tensor(torch.randn(3,3))]
    cases=[]
    def bad(i,value,error):
        args=base.copy();args[i]=value;cases.append((args,error))
    bad(0,[],TypeError)
    bad(0,Tensor(torch.randn(2,2,3,6,dtype=torch.float64)),TypeError)
    bad(0,Tensor(torch.randn(2,2,3,12)[...,::2]),ValueError)
    bad(0,Tensor(torch._neg_view(base[0].data)),ValueError)
    bad(0,Tensor(base[0].data,'cpu',False),ValueError)
    bad(1,Tensor(base[1].data,'cuda:0'),ValueError)
    bad(0,Tensor(torch.randn(2,2,3)),ValueError)
    bad(0,Tensor(torch.randn(2,0,3,6)),ValueError)
    bad(0,Tensor(torch.randn(2,2,3,5)),ValueError)
    bad(1,Tensor(torch.randn(1,3,3,6)),ValueError)
    bad(2,Tensor(torch.randn(3,4)),ValueError)
    bad(3,base[2],ValueError)
    for args,error in cases:
        events.clear()
        try: forward(*args)
        except error: pass
        else: raise AssertionError('unsupported input admitted')
        assert not any(x[0]=='launch' for x in events)
    # Sequence-one bug in the original: its singleton-axis stride is too small.
    original,_=host(ORIGINAL)
    q=Tensor(torch.arange(8,dtype=torch.float32).reshape(2,2,1,2))
    k=Tensor(torch.arange(8,dtype=torch.float32).reshape(2,2,1,2))
    c,s=Tensor(torch.full((1,1),2.)),Tensor(torch.full((1,1),3.))
    expected=reference(q.data.clone(),c.data,s.data)
    oq,*_=original(q,k,c,s)
    assert not torch.equal(oq.data,expected)
    print(f'Unchanged complete JIT AST; {len(shapes)} copy/reuse/empty forward paths; {len(cases)} rejection paths passed.')
    print('Original sequence-one row-stride counterexample reproduced. CPU replay only; CUDA mocked.')

if __name__=='__main__': check_host()
