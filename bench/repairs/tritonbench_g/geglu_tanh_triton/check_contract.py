"""Actual flat GEGLU source and in-place host replay on CPU backing storage."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'geglu_tanh_triton_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/geglu_tanh_triton/geglu_tanh_triton.py'
spec=importlib.util.spec_from_file_location('swiglu_direct_fixtures',HERE.parent/'swiglu_triton/check_contract.py')
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
Tensor,Pointer,storage,function=f.Tensor,f.Pointer,f.storage,f.function


def interpreter(path=SOURCE,repaired=True,commuted=False):
    pid=[0]
    def load(ptr,mask,other):
        value=torch.full(mask.shape,other,dtype=torch.float32)
        value[mask]=storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offset[mask]]
        return value
    def store(ptr,value,mask):
        storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offset[mask]]=value[mask]
    ns={'tl':SimpleNamespace(program_id=lambda _:torch.tensor(pid[0]),arange=torch.arange,
        int64=torch.int64,float32=torch.float32,load=load,store=store),'tanh':torch.tanh}
    names=['_geglu_tanh_forward_kernel','_geglu_tanh_backward_kernel']
    if repaired: names=[name+'_repaired' for name in names]
    nodes=[function(path,name) for name in names]
    for node in nodes:
        if commuted:
            assert repaired
            offset=node.body[1].value.left
            assert isinstance(offset,ast.BinOp) and isinstance(offset.op,ast.Mult)
            assert ast.unparse(offset)=='program_id * BLOCK_SIZE'
            offset.left,offset.right=offset.right,offset.left
        node.decorator_list=[]
        for arg in node.args.args: arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    return ns,pid


def host():
    kernels,pid=interpreter(); events=[]
    @contextlib.contextmanager
    def device(d): events.append(('device',d)); yield
    def empty_like(t):
        out=t.wrap(torch.empty_like(t.data)); events.append(('allocation',out)); return out
    class Launch:
        def __init__(self,name): self.name=name
        def __getitem__(self,grid):
            def run(x,y,z,n_elements,BLOCK_SIZE,num_warps,num_ctas):
                assert (BLOCK_SIZE,num_warps,num_ctas)==(128,4,1)
                events.append(('launch',self.name,grid,n_elements))
                for program in reversed(range(grid[0])):
                    pid[0]=program
                    kernels[self.name](*(Pointer(t.data) for t in (x,y,z)),n_elements,BLOCK_SIZE)
            return run
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=device)),
        'triton':SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b)}
    for name in ('_geglu_tanh_forward_kernel_repaired','_geglu_tanh_backward_kernel_repaired'): ns[name]=Launch(name)
    nodes=[function(SOURCE,name) for name in ('_check_tensor','_check_call','geglu_forward_repaired','geglu_backward_repaired')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns,events


def source_binding():
    for phase,first in (('forward','a_row'),('backward','dc_row')):
        original=function(ORIGINAL,f'_geglu_tanh_{phase}_kernel')
        repaired=function(SOURCE,f'_geglu_tanh_{phase}_kernel_repaired')
        def suffix(node):
            index=next(i for i,x in enumerate(node.body) if isinstance(x,ast.Assign)
                       and isinstance(x.targets[0],ast.Name) and x.targets[0].id==first)
            return [ast.dump(x) for x in node.body[index:]]
        assert suffix(original)==suffix(repaired)
        assert ast.unparse(repaired.body[1])=='col_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)'
        assert ast.unparse(repaired.body[2])=='mask = col_offsets < n_elements'
    calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n,ast.Call)]
    assert not any(c.endswith(('.contiguous','.clone','.copy_','.resolve_neg')) for c in calls)
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    contract=(HERE/'Contract.lean').read_text()
    for field in ('payloads','counts','forwardShape','sameDevice','positive','grid32','padded64','alignment','separate'):
        assert f'  {field} :' in contract


def independent(a,b,d):
    t=torch.tanh(0.7978845608028654*(a+0.044715*a*a*a))
    g=0.5*a*(1+t)
    derivative=0.5*(1+t)+0.5*a*(1-t*t)*0.7978845608028654*(1+0.134145*a*a)
    return g*b,d*b*derivative,d*g


def checks():
    ns,events=host(); forward,backward=ns['geglu_forward_repaired'],ns['geglu_backward_repaired']
    torch.manual_seed(61)
    shapes=[(3,1),(3,33),(2,127),(2,128),(2,129),(2,256),(2,3,257),(1,1025),(1,65537),(0,33)]
    for shape in shapes:
        n=torch.empty(shape).numel()
        ab=torch.randn(n+11); bb=torch.randn(n+13); db=torch.randn(n+15)
        a,b,dc=Tensor(ab[3:3+n].reshape(shape)),Tensor(bb[5:5+n].reshape(shape)),Tensor(db[7:7+n].reshape(shape))
        beforeA,beforeB,beforeD=ab.clone(),bb.clone(),db.clone()
        x,y,d=a.data.clone(),b.data.clone(),dc.data.clone()
        cf,daf,dbf=independent(x,y,d)
        events.clear()
        av,bv,c=forward(a,b)
        torch.testing.assert_close(c.data,cf,rtol=3e-6,atol=1e-6)
        torch.testing.assert_close(ab,beforeA,rtol=0,atol=0)
        torch.testing.assert_close(bb,beforeB,rtol=0,atol=0)
        assert tuple(av.shape)==(n//shape[-1],shape[-1]) and bv.shape==av.shape
        assert av.untyped_storage().data_ptr()==a.untyped_storage().data_ptr()
        assert bv.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
        assert tuple(c.shape)==shape and c is not a and c is not b
        assert sum(e[0]=='launch' for e in events)==bool(n)
        events.clear()
        da,dbout=backward(av,bv,dc)
        torch.testing.assert_close(da.data,daf,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(dbout.data,dbf,rtol=3e-6,atol=1e-6)
        assert tuple(da.shape)==shape and tuple(dbout.shape)==shape
        assert da.untyped_storage().data_ptr()==a.untyped_storage().data_ptr()
        assert dbout.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
        torch.testing.assert_close(db,beforeD,rtol=0,atol=0)
        assert torch.equal(ab[:3],beforeA[:3]) and torch.equal(ab[3+n:],beforeA[3+n:])
        assert torch.equal(bb[:5],beforeB[:5]) and torch.equal(bb[5+n:],beforeB[5+n:])
        assert sum(e[0]=='launch' for e in events)==bool(n)
    a,b=Tensor(torch.randn(2,8)),Tensor(torch.randn(2,8))
    failures=[([],b,TypeError),(Tensor(torch.randn(2,8,dtype=torch.float64)),b,TypeError),
        (Tensor(torch.randn(2,16)[:,::2]),b,ValueError),(Tensor(torch._neg_view(a.data)),b,ValueError),
        (Tensor(a.data,'cpu',False),b,ValueError),(a,Tensor(b.data,'cuda:0'),ValueError),
        (a,Tensor(torch.randn(4,4)),ValueError),(a,a,ValueError),
        (Tensor(torch.empty(1<<31,128,device='meta')),Tensor(torch.empty(1<<31,128,device='meta')),ValueError),
        (Tensor(torch.empty(2,0)),Tensor(torch.empty(2,0)),ValueError)]
    for offset in (1,2,3):
        bytes_=torch.arange(40,dtype=torch.uint8).untyped_storage()[offset:offset+32]
        unaligned=torch.empty(0).set_(bytes_,0,(2,4),(4,1))
        failures.append((Tensor(unaligned),Tensor(torch.randn(2,4)),ValueError))
    for x,y,error in failures:
        events.clear()
        try: forward(x,y)
        except error: pass
        else: raise AssertionError('unsupported input admitted')
        assert not any(e[0]=='launch' for e in events)
    try: backward(a,b,a)
    except ValueError: pass
    else: raise AssertionError('upstream/output overlap admitted')
    print(f'Original scalar AST retained; {len(shapes)} forward/in-place-backward paths; {len(failures)+1} rejections passed.')
    print(f'CPU physical-storage replay only; CUDA mocked. CUDA available: {torch.cuda.is_available()}.')


def original_counterexample():
    kernels,pid=interpreter(ORIGINAL,False)
    a=torch.full((2,256),0.7); b=torch.full_like(a,0.9); c=torch.full_like(a,-91.)
    for row in range(2):
        pid[0]=row
        kernels['_geglu_tanh_forward_kernel'](Pointer(a),Pointer(b),Pointer(c),256,256,128)
    assert torch.all(c[:,128:]==-91.)
    assert not torch.allclose(c,independent(a,b,torch.ones_like(a))[0])
    print('Original width-256 missing-column counterexample reproduced.')


def index_variant_checks():
    original,pid=interpreter()
    commuted,other_pid=interpreter(commuted=True)
    counts=(0,1,127,128,129,256,517)
    edge_words=torch.tensor([0,0x80000000,1,0x007fffff,0x7f800000,0xff800000,0x7fc01234],dtype=torch.uint32).view(torch.float32)
    for n in counts:
        for backward in (False,True):
            arrays=[torch.full((n+11,),-91.,dtype=torch.float32) for _ in range(3)]
            for k,array in enumerate(arrays):
                if n:
                    array[3:3+n]=torch.linspace(-2+k,2+k,n)
                    array[3:3+min(n,len(edge_words))]=edge_words[:min(n,len(edge_words))]
            copies=[a.clone() for a in arrays]
            name='_geglu_tanh_backward_kernel_repaired' if backward else '_geglu_tanh_forward_kernel_repaired'
            for program in reversed(range((n+127)//128)):
                pid[0]=other_pid[0]=program
                original[name](*(Pointer(a[3:3+n]) for a in arrays),n,128)
                commuted[name](*(Pointer(a[3:3+n]) for a in copies),n,128)
            for a,b in zip(arrays,copies):
                assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                assert torch.all(a[:3]==-91.) and torch.all(a[3+n:]==-91.)
    print(f'{2*len(counts)} actual index-commuted source AST cases passed with exact backing bytes and exceptional FP32 words.')

if __name__=='__main__':
    source_binding(); checks(); original_counterexample(); index_variant_checks()
