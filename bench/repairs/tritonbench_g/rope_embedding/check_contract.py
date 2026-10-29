"""Original clone/reshape/squeeze RoPE paths with CPU JIT storage replay."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'rope_embedding_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rope_embedding/rope_embedding.py'
spec=importlib.util.spec_from_file_location('rope_backward_fixtures',HERE.parent/'rope_backward_transform/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
Pointer,storage,function=f.Pointer,f.storage,f.function
EVENTS=[]

class Tensor(f.Tensor):
    def wrap(self,value):return Tensor(value,self.device,self.is_cuda)
    def clone(self,*args,**kwargs):
        EVENTS.append(('clone',args,kwargs));return self.wrap(self.data.clone(*args,**kwargs))
    def squeeze(self):
        out=self.wrap(self.data.squeeze());EVENTS.append(('squeeze',self,out));return out
    def view(self,*shape):return self.wrap(self.data.view(*shape))


def host():
    kernel,ids=f.f.f.cpu_kernel(SOURCE,'_rope_embedding')
    @contextlib.contextmanager
    def device(d):EVENTS.append(('device',d));yield
    class Launch:
        def __getitem__(self,grid):
            def run(Q,qs,c,cs,s,ss,*args,**kwargs):
                EVENTS.append(('launch',grid,qs));kwargs.pop('num_warps')
                for row in reversed(range(grid[0])):
                    ids[0]=row
                    for group in reversed(range(grid[1])):
                        ids[1]=group
                        kernel(Pointer(Q.data),qs,Pointer(c.data),cs,Pointer(s.data),ss,*args,**kwargs)
            return run
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,cuda=SimpleNamespace(device=device)),
        'next_power_of_2':lambda n:1<<(n-1).bit_length(),'MAX_FUSED_SIZE':65536,'ROPE_GROUP_SIZE':4,'_rope_embedding':Launch()}
    names=('calculate_settings','_check_payload','_check_tables','_check_alias','_geometry','_check_pointers',
           '_rope_embedding_forward_impl','_rope_embedding_backward_impl')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function(SOURCE,name) for name in names],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns


def reference(x,c,s,back):
    half=x.shape[-1]//2;seq=x.shape[2]
    c,s=c.reshape(-1,half)[:seq] if half else c,s.reshape(-1,half)[:seq] if half else s
    if half==0:return x.clone()
    a,b=x[...,:half],x[...,half:2*half]
    return torch.cat((a*c+b*s if back else a*c-b*s,b*c-a*s if back else b*c+a*s,x[...,2*half:]),-1)


def check_source():
    for name in ('calculate_settings','_rope_embedding'):
        assert ast.dump(function(SOURCE,name))==ast.dump(function(ORIGINAL,name))
    forward=function(SOURCE,'_rope_embedding_forward_impl');backward=function(SOURCE,'_rope_embedding_backward_impl')
    clone=[ast.unparse(n) for n in ast.walk(forward) if isinstance(n,ast.Call) and ast.unparse(n.func).endswith('.clone')]
    assert clone==['Q.transpose(1, 2).clone()']
    calls=[ast.unparse(n.func) for n in ast.walk(backward) if isinstance(n,ast.Call)]
    assert not any(c.endswith(('.clone','.contiguous','.copy_','.resolve_neg','.to')) for c in calls)


def check_host():
    ns=host();forward=ns['_rope_embedding_forward_impl'];backward=ns['_rope_embedding_backward_impl']
    torch.manual_seed(83)
    shapes=[(2,7,3,6),(2,3,1,6),(2,1,3,5),(1,5,2,2),(1,1,1,2),
            (0,3,4,6),(2,3,0,6),(2,0,3,6),(2,3,4,1),(2,3,4,0),(1,5,3,34)]
    for shape in shapes:
        batch,heads,seq,dim=shape;n=torch.empty(shape).numel();half=dim//2
        backing=torch.randn(n+9);q=Tensor(backing[3:3+n].reshape(shape));old=backing.clone();initial=q.data.clone()
        rows=max(seq,2)
        cos=Tensor(torch.randn(1,rows,1,half));sin=Tensor(torch.randn(1,rows,1,half))
        EVENTS.clear();out,c,s,groups,block,warps=forward(q,cos,sin)
        torch.testing.assert_close(out.data,reference(initial,c.data,s.data,False),rtol=0,atol=0)
        assert torch.equal(backing,old)
        assert sum(e[0]=='clone' for e in EVENTS)==1
        assert next(e for e in EVENTS if e[0]=='clone')[1:]==((),{})
        sq=[e for e in EVENTS if e[0]=='squeeze'];assert c is sq[0][2] and s is sq[1][2]
        assert c.untyped_storage()._cdata==cos.untyped_storage()._cdata
        assert s.untyped_storage()._cdata==sin.untyped_storage()._cdata
        assert out.untyped_storage()._cdata!=q.untyped_storage()._cdata
        assert sum(e[0]=='launch' for e in EVENTS)==bool(n and dim>=2)
        gradraw=torch.randn(n+11);grad=Tensor(gradraw[4:4+n].reshape(shape));g0=grad.data.clone();gbefore=gradraw.clone()
        EVENTS.clear();dout=backward(grad,c,s,groups,block,warps)
        torch.testing.assert_close(dout.data,reference(g0,c.data,s.data,True),rtol=0,atol=0)
        reuse=n==0 or seq<=1 or heads<=1
        assert (dout.untyped_storage()._cdata==grad.untyped_storage()._cdata)==reuse
        assert not any(e[0] in ('clone','squeeze') for e in EVENTS)
        if not reuse:assert torch.equal(gradraw,gbefore)
        assert torch.equal(gradraw[:4],gbefore[:4]) and torch.equal(gradraw[4+n:],gbefore[4+n:])
        assert sum(e[0]=='launch' for e in EVENTS)==bool(n and dim>=2)
    q,c,s=Tensor(torch.randn(2,3,4,6)),Tensor(torch.randn(4,3)),Tensor(torch.randn(4,3))
    failures=[([],c,s,TypeError),(Tensor(torch.randn(2,3,4,6,dtype=torch.float64)),c,s,TypeError),
        (Tensor(torch.randn(2,3,4,12)[...,::2]),c,s,ValueError),(Tensor(torch._neg_view(q.data)),c,s,ValueError),
        (Tensor(q.data,'cpu',False),c,s,ValueError),(q,Tensor(c.data,'cuda:0'),s,ValueError),
        (Tensor(torch.randn(3,4,6)),c,s,ValueError),(q,Tensor(torch.randn(3,3)),s,ValueError),
        (q,Tensor(torch.randn(4,4)),s,ValueError),
        (Tensor(torch.randn(1,1,1,6)),Tensor(torch.randn(1,3)),Tensor(torch.randn(1,3)),ValueError),
        (Tensor(torch.randn(1,1,1,2)),Tensor(torch.randn(1,1)),Tensor(torch.randn(1,1)),ValueError)]
    for x,cos,sin,error in failures:
        EVENTS.clear()
        try:forward(x,cos,sin)
        except error:pass
        else:raise AssertionError('unsupported input admitted')
        assert not any(e[0]=='launch' for e in EVENTS)
    for cache in ((2,4,4),(1,8,4),(1,4,8)):
        try:backward(q,c,s,*cache)
        except ValueError:pass
        else:raise AssertionError('bad cache admitted')
    alias=Tensor(q.data.flatten()[:12].view(4,3))
    try:backward(q,alias,s,1,4,4)
    except ValueError:pass
    else:raise AssertionError('writable/coefficient alias admitted')
    print(f'Unchanged JIT/settings AST; {len(shapes)} original forward clone and backward reshape paths; {len(failures)+4} rejections passed.')
    print('Actual squeeze-view objects, odd tails, byte-prefix/suffix frame and cache validation checked. CPU only; CUDA mocked.')

def check_original_defect():
    """The imported wrapper checks cosine length only; short sine is unsafe."""
    original_forward=function(ORIGINAL,'_rope_embedding_forward_impl')
    assertions=[ast.unparse(n.test) for n in ast.walk(original_forward) if isinstance(n,ast.Assert)]
    assert assertions==['seq_len <= cos.shape[0]']
    kernel,ids=f.f.f.cpu_kernel(ORIGINAL,'_rope_embedding')
    q=torch.arange(24,dtype=torch.float32).reshape(3,8)
    cosine=torch.ones(3,2)
    sine=torch.ones(1,2).squeeze()
    ids[0],ids[1]=2,0
    try:
        kernel(Pointer(q),8,Pointer(cosine),2,Pointer(sine),1,3,4,2,
               BACKWARD_PASS=False,BLOCK_SIZE=2)
    except IndexError:
        pass
    else:
        raise AssertionError('expected the original short-sine out-of-bounds read')
    # The checked source rejects this before submitting a kernel.
    ns=host();EVENTS.clear()
    try:
        ns['_rope_embedding_forward_impl'](Tensor(torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)),
                                            Tensor(cosine),Tensor(torch.ones(1,2)))
    except ValueError:
        pass
    else:
        raise AssertionError('short sine unexpectedly admitted')
    assert not any(e[0]=='launch' for e in EVENTS)
    print('Original counterexample: B=1,H=2,S=3,D=4, cosine [3,2], sine [1,2]; row 2 reads past sine storage. Checked repair rejects.')


if __name__=='__main__':check_source();check_host();check_original_defect()
