"""Actual fast RoPE methods and sequential wrapper replay; CUDA/autograd dispatch mocked."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'fast_rope_embedding_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/fast_rope_embedding/fast_rope_embedding.py'
spec=importlib.util.spec_from_file_location('embedding_fixtures',HERE.parent/'rope_embedding/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
Tensor,Pointer,storage,function,EVENTS=f.Tensor,f.Pointer,f.storage,f.function,f.EVENTS


def host():
    kernel,ids=f.f.f.f.cpu_kernel(SOURCE,'_rope_embedding')
    contexts=[]
    @contextlib.contextmanager
    def device(d):EVENTS.append(('device',d));yield
    class Function:
        @classmethod
        def apply(cls,*args):
            ctx=SimpleNamespace();contexts.append(ctx);return cls.forward(ctx,*args)
    class Launch:
        def __getitem__(self,grid):
            def run(Q,qs,c,cs,s,ss,*args,**kwargs):
                EVENTS.append(('launch',grid,kwargs['BACKWARD_PASS']));kwargs.pop('num_warps')
                for row in reversed(range(grid[0])):
                    ids[0]=row
                    for group in reversed(range(grid[1])):
                        ids[1]=group
                        kernel(Pointer(Q.data),qs,Pointer(c.data),cs,Pointer(s.data),ss,*args,**kwargs)
            return run
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        cuda=SimpleNamespace(device=device),autograd=SimpleNamespace(Function=Function)),
        'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),
        'MAX_FUSED_SIZE':65536,'ROPE_GROUP_SIZE':4,'_rope_embedding':Launch()}
    names=('calculate_settings','_check_payload','_check_tables','_check_alias','_geometry','_check_pointers','fast_rope_embedding')
    nodes=[function(SOURCE,name) for name in names]
    nodes.insert(-1,next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns,contexts


def check_source():
    for name in ('calculate_settings','_rope_embedding'):
        assert ast.dump(function(SOURCE,name))==ast.dump(function(ORIGINAL,name))
    original=function(ORIGINAL,'fast_rope_embedding')
    repaired=function(SOURCE,'fast_rope_embedding')
    for statement in original.body:
        assert any(ast.dump(statement)==ast.dump(s) for s in repaired.body)
    source=ast.parse(SOURCE.read_text())
    calls=[ast.unparse(n.func) for n in ast.walk(source) if isinstance(n,ast.Call)]
    assert not any(name.endswith(('.clone','.contiguous','.copy_','.resolve_neg','.save_for_backward')) for name in calls)
    assert not any('_version' in ast.unparse(n) for n in ast.walk(source) if isinstance(n,ast.Attribute))


def check_host():
    ns,contexts=host();forward=ns['fast_rope_embedding'];callback=ns['Fast_RoPE_Embedding'].backward
    torch.manual_seed(89)
    shapes=[(2,7,3,4,6),(2,3,2,1,6),(2,1,3,4,5),(1,5,1,2,2),
            (1,1,1,1,2),(0,3,2,4,6),(2,3,2,0,6),(2,0,2,3,6),
            (2,3,2,4,1),(2,3,2,4,0),(1,5,3,3,34)]
    for batch,hq,hk,seq,dim in shapes:
        inputs=[];backing=[];before=[]
        for heads in (hq,hk):
            n=batch*heads*seq*dim;raw=torch.randn(n+11)
            inputs.append(Tensor(raw[3:3+n].view(batch,heads,seq,dim)));backing.append(raw);before.append(raw.clone())
        initial=[x.data.clone() for x in inputs]
        c,s=Tensor(torch.randn(1,max(seq,2),1,dim//2)),Tensor(torch.randn(1,max(seq,2),1,dim//2))
        EVENTS.clear();contexts.clear();qout,kout=forward(*inputs,c,s)
        assert len(contexts)==2
        assert not any(e[0]=='clone' for e in EVENTS)
        for inp,out,x,raw,old in zip(inputs,(qout,kout),initial,backing,before):
            torch.testing.assert_close(out.data,f.reference(x,c.data.squeeze(),s.data.squeeze(),False),rtol=0,atol=0)
            reuse=not inp.numel() or seq<=1 or inp.shape[1]<=1
            assert (out.untyped_storage()._cdata==inp.untyped_storage()._cdata)==reuse
            if not reuse:assert torch.equal(raw,old)
            assert torch.equal(raw[:3],old[:3]) and torch.equal(raw[3+inp.numel():],old[3+inp.numel():])
        for ctx in contexts:
            assert set(vars(ctx))=={'BLOCK_SIZE','num_warps','n_groups','cos','sin'}
            assert ctx.cos.untyped_storage()._cdata==c.untyped_storage()._cdata
            assert ctx.sin.untyped_storage()._cdata==s.untyped_storage()._cdata
        # Plain cached coefficient views intentionally observe later mutations.
        c.data.mul_(2);s.data.add_(0.25)
        for heads,ctx in zip((hq,hk),contexts):
            n=batch*heads*seq*dim;raw=torch.randn(n+13);grad=Tensor(raw[5:5+n].view(batch,heads,seq,dim))
            old=raw.clone();initial=grad.data.clone();EVENTS.clear()
            internal,dc,ds=callback(ctx,grad.transpose(1,2));out=internal.transpose(1,2)
            assert dc is None and ds is None
            assert tuple(internal.shape)==(batch,seq,heads,dim)
            torch.testing.assert_close(out.data,f.reference(initial,ctx.cos.data,ctx.sin.data,True),rtol=0,atol=0)
            reuse=not n or seq<=1 or heads<=1
            assert (out.untyped_storage()._cdata==grad.untyped_storage()._cdata)==reuse
            if not reuse:assert torch.equal(raw,old)
            assert torch.equal(raw[:5],old[:5]) and torch.equal(raw[5+n:],old[5+n:])
            assert not any(e[0] in ('clone','squeeze') for e in EVENTS)
    q,k=Tensor(torch.randn(2,3,4,6)),Tensor(torch.randn(2,2,4,6))
    c,s=Tensor(torch.randn(4,3)),Tensor(torch.randn(4,3))
    failures=[([],k,c,s,TypeError),(q,Tensor(torch.randn(2,2,4,6,dtype=torch.float64)),c,s,TypeError),
        (Tensor(torch.randn(2,3,4,12)[...,::2]),k,c,s,ValueError),
        (Tensor(torch._neg_view(q.data)),k,c,s,ValueError),(Tensor(q.data,'cpu',False),k,c,s,ValueError),
        (q,Tensor(k.data,'cuda:0'),c,s,ValueError),(q,Tensor(torch.randn(1,2,4,6)),c,s,ValueError),
        (q,q,c,s,ValueError),(q,k,Tensor(torch.randn(3,3)),s,ValueError),
        (q,k,Tensor(q.data.flatten()[:12].view(4,3)),s,ValueError)]
    for a,b,cos,sin,error in failures:
        EVENTS.clear()
        try:forward(a,b,cos,sin)
        except error:pass
        else:raise AssertionError('unsupported fast input admitted')
        assert not any(e[0]=='launch' for e in EVENTS)
    contexts.clear();forward(q,k,c,s);ctx=contexts[0];ctx.BLOCK_SIZE*=2
    try:callback(ctx,Tensor(torch.randn_like(q.data)).transpose(1,2))
    except ValueError:pass
    else:raise AssertionError('bad cached geometry admitted')
    print(f'Unchanged JIT/settings and outer apply ASTs; {len(shapes)} sequential Q/K forward plus both backward callback paths; {len(failures)+1} rejections passed.')
    print('Current cached coefficient mutation, exact None returns, reshape reuse and backing frame verified. CPU replay; CUDA/autograd dispatch mocked.')

if __name__=='__main__':check_source();check_host()
