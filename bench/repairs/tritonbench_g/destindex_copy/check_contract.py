"""Dual-plane actual AST checks on CPU storage; CUDA context/launch mocked."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'destindex_copy_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/destindex_copy/destindex_copy.py'
COMMON = HERE.parent / 'destindex_copy_kv1/check_contract.py'
spec = importlib.util.spec_from_file_location('kv1_test_helpers', COMMON)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
Tensor, Pointer = helpers.Tensor, helpers.Pointer
backing_bytes, function = helpers.backing_bytes, helpers.function


def source_binding():
    selected = function(SOURCE, '_copy_heads')
    reference = function(HERE.parents[2] / 'tritonbench_g/destindex_copy_kv1/destindex_copy_kv1.py',
                         '_fwd_kernel_destindex_copy_kv')
    selected.name = reference.name
    assert ast.dump(selected, include_attributes=False) == ast.dump(reference, include_attributes=False)
    wrapper = function(SOURCE, 'destindex_copy_kv_repaired')
    guards = [n for n in wrapper.body if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body)]
    assert len(guards)==15
    forbidden={'clone','contiguous','resolve_neg','copy_','to'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in forbidden for n in ast.walk(wrapper))
    contract=(HERE/'Contract.lean').read_text()
    assert 'common : ∀ s' in contract and 'separated : Separated m' in contract
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    assert 'FPEquiv' not in (HERE/'Correct.lean').read_text()
    assert 'Correct' not in (HERE/'FPEquiv.lean').read_text()


def main():
    source_binding()
    spec=importlib.util.spec_from_file_location('dual_copy_repair',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    helpers.NAME='_copy_heads'
    kernel,pid=helpers.interpreter(SOURCE)
    calls=[]
    class Launch:
        def __getitem__(self,grid):
            def run(k,dest,out,*args,**kw):
                calls.append((grid,k,dest,out,args,kw))
                for p in reversed(range(grid[0])):
                    pid[0]=p
                    kernel(Pointer(k.data),Pointer(dest.data),Pointer(out.data),*args,
                           kw['BLOCK_DMODEL'],kw['BLOCK_HEAD'])
            return run
    module._copy_heads=Launch()
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int32=torch.int32,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    run=module.destindex_copy_kv_repaired
    cases=[(0,3,5,2,7),(2,0,3,2,5),(2,3,0,0,5),(1,1,1,1,1),
           (3,3,5,2,7),(2,12,128,1,64),(2,12,96,3,33),(2,3,257,2,129),(2,3,4,3,4)]
    for count,h0,d0,h1,d1 in cases:
        src=[];out=[]
        for h,d in [(h0,d0),(h1,d1)]:
            size=(count+1)*h*d
            bits=torch.arange(size+8,dtype=torch.int32)*16381
            bits[:6]=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1],dtype=torch.int32)
            src.append(Tensor(bits.view(torch.float32)[1:1+size].reshape(count+1,h,d)))
            osize=(count+3)*h*d
            out.append(Tensor(torch.full((osize+10,),42.)[3:3+osize].reshape(count+3,h,d)))
        if (h0,d0,h1,d1)==(3,4,3,4): src[1]=src[0]
        dest=Tensor(torch.arange(count,0,-1,dtype=torch.int32))
        args=[src[0],src[1],dest,out[0],out[1]]
        snapshots=[backing_bytes(t.data).clone() for t in args]
        expected=[t.data.view(torch.int32).clone() for t in out]
        for side in range(2):
            for row,loc in enumerate(dest.data.tolist()): expected[side][loc]=src[side].data.view(torch.int32)[row]
        old=len(calls)
        assert run(*args) is None
        assert len(calls)-old==int(count>0 and h0>0 and d0>0)+int(count>0 and h1>0 and d1>0)
        for side in range(2):
            tensor=out[side].data
            assert torch.equal(tensor.view(torch.int32),expected[side])
            after=backing_bytes(tensor);before=snapshots[3+side]
            lo,hi=tensor.storage_offset()*4,(tensor.storage_offset()+tensor.numel())*4
            assert torch.equal(after[:lo],before[:lo]) and torch.equal(after[hi:],before[hi:])
        assert all(torch.equal(backing_bytes(args[i].data),snapshots[i]) for i in range(3))
        for call in calls[old:]:
            assert call[2] is dest
            assert (call[1] is src[0] and call[3] is out[0]) or (call[1] is src[1] and call[3] is out[1])

    def fresh():
        return [Tensor(torch.arange(24.).reshape(2,3,4)),Tensor(torch.arange(20.).reshape(2,2,5)),
                Tensor(torch.tensor([2,0],dtype=torch.int32)),Tensor(torch.full((3,3,4),42.)),Tensor(torch.full((3,2,5),43.))]
    rejected=0
    def reject(args):
        nonlocal rejected
        old=len(calls); ts=[t for t in args if isinstance(t,Tensor)]
        snapshots=[backing_bytes(t.data).clone() for t in ts]
        try:run(*args)
        except (ValueError,TypeError):pass
        else:raise AssertionError('unsupported arguments admitted')
        assert len(calls)==old
        assert all(torch.equal(s,backing_bytes(t.data)) for s,t in zip(snapshots,ts))
        rejected+=1
    a=fresh();a[0]=None;reject(a)
    for i in range(5):
        a=fresh();a[i].is_cuda=False;reject(a)
        a=fresh();a[i].device=torch.device('cuda:1');reject(a)
        a=fresh();a[i]=Tensor(a[i].data.to(torch.float64));reject(a)
        a=fresh();a[i].data_ptr=lambda:3;reject(a)
        a=fresh();a[i].numel=lambda:1<<31;reject(a)
    for i in [0,1,3,4]:
        a=fresh();a[i]=Tensor(torch._neg_view(a[i].data));reject(a)
        a=fresh();a[i]=Tensor(a[i].data.reshape(-1));reject(a)
    a=fresh();a[2]=Tensor(torch.tensor([0,1,2,0],dtype=torch.int32)[::2]);reject(a)
    a=fresh();a[3]=Tensor(torch.ones(3,3,8)[:,:,::2]);reject(a)
    a=fresh();a[1]=Tensor(torch.ones(2,2,10)[:,:,::2]);reject(a)
    a=fresh();a[3]=Tensor(torch.ones(3,3,5));reject(a)
    a=fresh();a[4]=Tensor(torch.ones(3,3,5));reject(a)
    for values in [[-1,0],[3,0],[1,1],[0,1,2]]:
        a=fresh();a[2]=Tensor(torch.tensor(values,dtype=torch.int32));reject(a)
    # Own-output, cross-plane input/output, and output/output aliases are rejected.
    a=fresh();a[3]=a[0];a[2]=Tensor(torch.tensor([0,1],dtype=torch.int32));reject(a)
    a=fresh();a[1]=Tensor(torch.ones(2,3,4));a[4]=Tensor(torch.ones(3,3,4));a[3]=a[1];a[2]=Tensor(torch.tensor([0,1],dtype=torch.int32));reject(a)
    a=fresh();a[1]=Tensor(torch.ones(2,3,4));a[4]=a[3];reject(a)
    a=fresh();a[0]=Tensor(torch.empty(0,1025,1025));a[3]=Tensor(torch.empty(0,1025,1025));a[2]=Tensor(torch.empty(0,dtype=torch.int32));reject(a)

    helpers.NAME='_fwd_kernel_destindex_copy_kv'
    original,opid=helpers.interpreter(ORIGINAL)
    def original_run(k0,k1,dest,o0,o1,count):
        for p in range(count):
            opid[0]=p
            args=[]
            for t in [k0,k1,o0,o1]:args.extend([t.shape[1]*t.shape[2],t.shape[2],1])
            original(*(Pointer(t) for t in [k0,k1,dest,o0,o1]),*args,k0.shape[1],k1.shape[1],
                1<<(k0.shape[2]-1).bit_length(),1<<(k1.shape[2]-1).bit_length())
    k0=torch.arange(48.).reshape(1,12,4);k1=torch.ones(1,1,4)
    o0=torch.full_like(k0,-1.);o1=torch.full_like(k1,-1.)
    original_run(k0,k1,torch.tensor([0],dtype=torch.int32),o0,o1,1)
    assert torch.equal(o0[:,0],k0[:,0]) and bool((o0[:,1:]==-1).all())
    # Rounded width three writes the exterior fourth word in both outputs.
    k0=torch.tensor([1.,2.,3.,91.])[:3].reshape(1,1,3);k1=torch.tensor([4.,5.,6.,92.])[:3].reshape(1,1,3)
    backing0=torch.tensor([0.,0.,0.,-71.]);backing1=torch.tensor([0.,0.,0.,-72.])
    original_run(k0,k1,torch.tensor([0],dtype=torch.int32),backing0[:3].reshape(1,1,3),backing1[:3].reshape(1,1,3),1)
    assert backing0[3]==91. and backing1[3]==92.
    k0=torch.tensor([11.,22.]).reshape(2,1,1);k1=torch.tensor([33.,44.]).reshape(2,1,1)
    o0=torch.tensor([100.,200.,300.]).reshape(3,1,1);o1=o0.clone()
    original_run(k0,k1,torch.tensor([0,1,2],dtype=torch.int32)[::2],o0,o1,2)
    assert o0.flatten().tolist()==[11.,22.,300.] and o1.flatten().tolist()==[33.,44.,300.]
    print(f'Masked multihead source AST, {len(cases)} admissions, {rejected} rejections, and all 3 original failures passed.')
    print('Raw FP32 bits and both caller byte frames checked on CPU; CUDA mocked; no GPU claim.')


if __name__=='__main__':main()
