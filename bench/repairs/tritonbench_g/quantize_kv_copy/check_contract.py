"""Execute the actual JIT AST on CPU storage; CUDA device and launch are mocked."""
import ast
import contextlib
import importlib.util
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'quantize_kv_copy_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/quantize_kv_copy/quantize_kv_copy.py'
ACCEPTED_AST_SHA256 = '2c8270da01af4c1cd41b64b03441b9108bd22619a178efdd90dbc93ee906b721'
NAME = '_fwd_kernel_destindex_copy_quantize_kv'
spec = importlib.util.spec_from_file_location('copy_test_helpers', HERE.parent / 'destindex_copy_kv1/check_contract.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
function, raw, backing_bytes = helpers.function, helpers.raw, helpers.backing_bytes

class Pointer(helpers.Pointer):
    @property
    def dtype(self): return SimpleNamespace(element_ty=self.data.dtype)
    def __add__(self, offset): return Pointer(self.data,self.offset+offset)


class Tensor(helpers.Tensor):
    def element_size(self): return self.data.element_size()
    def view(self,*shape): return Tensor(self.data.view(*shape),str(self.device),self.is_cuda)


def source_binding():
    assert hashlib.sha256(ast.dump(function(SOURCE, NAME), include_attributes=False).encode()).hexdigest() == ACCEPTED_AST_SHA256
    wrapper = function(SOURCE, 'destindex_copy_quantize_kv')
    guards = [n for n in wrapper.body if isinstance(n, ast.If) and any(isinstance(s, ast.Raise) for s in n.body)]
    assert len(guards) == 17
    lean = (HERE / 'Contract.lean').read_text().split('def checkInputs', 1)[1].split('def checkArguments', 1)[0]
    assert re.findall(r'\.error \.(\w+)', lean) == ['layout','dtype','cuda','device','rank','contiguous','negative','width','shapes','grid','extent','tile','alignment','overlap','bounds','unique']
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'clone','contiguous','resolve_neg','copy_','to'} for n in ast.walk(wrapper))
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()
    assert 'FPEquiv' not in (HERE / 'Correct.lean').read_text()
    assert 'Correct' not in (HERE / 'FPEquiv.lean').read_text()


def interpreter(path):
    pid = [0,0]
    def load(pointer, mask=None, other=0.0):
        storage = raw(pointer.data)
        offsets = torch.as_tensor(pointer.offset + pointer.data.storage_offset())
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        mask = torch.broadcast_to(mask, offsets.shape)
        selected = offsets[mask].long()
        assert bool(((selected >= 0) & (selected < len(storage))).all()), 'source OOB'
        result = torch.full_like(offsets, other, dtype=pointer.data.dtype)
        result[mask] = storage[selected]
        return result
    def store(pointer, values, mask=None):
        storage = raw(pointer.data)
        offsets = torch.as_tensor(pointer.offset)
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        mask = torch.broadcast_to(mask, offsets.shape)
        indices = pointer.data.storage_offset() + offsets[mask].long()
        assert bool(((indices >= 0) & (indices < len(storage))).all()), 'output OOB'
        storage[indices] = values[mask]
    node = function(path, NAME)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    ns = {'tl': SimpleNamespace(program_id=lambda axis: torch.tensor(pid[axis], dtype=torch.int32),
        arange=lambda a,b: torch.arange(a,b,dtype=torch.int32),load=load,store=store,
        int32=torch.int32,int64=torch.int64,int8=torch.int8,float16=torch.float16,float32=torch.float32,
        abs=torch.abs,max=lambda x,axis: torch.max(x,dim=axis).values,where=torch.where)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    return ns[NAME], pid


def main():
    source_binding()
    spec = importlib.util.spec_from_file_location('direct_quantize', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel,pid = interpreter(SOURCE)
    calls = []
    class Launch:
        def __getitem__(self,grid):
            def run(k,dest,out,scale,*args,**kw):
                calls.append((grid,k,dest,out,scale,args,kw))
                for p in reversed(range(grid[0])):
                    for head in reversed(range(grid[1])):
                        pid[:]=[p,head]
                        kernel(*(Pointer(t.data) for t in (k,dest,out,scale)),*args,kw['BLOCK_GROUP_NUM'],kw['BLOCK_GROUP_DIM'])
            return run
    module._fwd_kernel_destindex_copy_quantize_kv=Launch()
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        int32=torch.int32,int64=torch.int64,int8=torch.int8,float16=torch.float16,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    run=module.destindex_copy_quantize_kv
    cases=[(0,3,8),(1,1,8),(2,3,16),(3,3,24),(2,7,40),(2,12,64),(1,5,256),(2,3,8)]
    for count,heads,dims in cases:
        size=(count+1)*heads*dims
        src=torch.arange(size+6,dtype=torch.float32)[1:1+size].reshape(count+1,heads,dims)
        src.copy_((src%255)-127)
        src.reshape(count+1,heads,dims//8,8)[:,:,:,-1]=127
        src[::2]*=2
        if count: src[0,0]=0
        output=torch.full(((count+3)*heads*dims+10,),-7,dtype=torch.int8)[3:3+(count+3)*heads*dims].reshape(count+3,heads,dims)
        scale=torch.full(((count+3)*heads*(dims//8)+6,),7.,dtype=torch.float32)[2:2+(count+3)*heads*(dims//8)].reshape(count+3,heads,dims//8)
        dest=torch.arange(count+2,dtype=torch.int32)[1:count+1].flip(0)
        args=[Tensor(t) for t in (src,dest,output,scale)]
        snapshots=[backing_bytes(t).clone() for t in (src,dest,output,scale)]
        expected,expected_scale=output.clone(),scale.clone()
        for s,d in enumerate(dest.tolist()):
            grouped=src[s].reshape(heads,dims//8,8)
            maximum=grouped.abs().amax(2,keepdim=True)
            row_scale=maximum/127
            expected[d]=(grouped/torch.where(maximum==0,1.,row_scale)).to(torch.int8).reshape(heads,dims)
            expected_scale[d]=row_scale[:,:,0]
        old_calls=len(calls)
        assert run(*args) is None
        assert torch.equal(output,expected) and torch.equal(scale,expected_scale)
        for t,snapshot in zip((src,dest,output,scale),snapshots):
            if t is src or t is dest: assert torch.equal(backing_bytes(t),snapshot)
            start=t.storage_offset()*t.element_size();end=start+t.numel()*t.element_size()
            assert torch.equal(backing_bytes(t)[:start],snapshot[:start]) and torch.equal(backing_bytes(t)[end:],snapshot[end:])
        assert len(calls)-old_calls==int(count>0)
        if count:
            assert calls[-1][2] is args[1] and calls[-1][4] is args[3]
            for actual,original in [(calls[-1][1],args[0]),(calls[-1][3],args[2])]:
                assert actual.data_ptr()==original.data_ptr() and actual.data.untyped_storage()._cdata==original.data.untyped_storage()._cdata
                assert actual.shape==(*original.shape[:2],dims//8,8)
    # The guard admits every FP32 pattern. This checks admission and byte framing,
    # not a GPU-defined value for NaN, infinity or out-of-range integer casts.
    bits=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1,1065353216,-1082130432],dtype=torch.int32)
    patterned=[Tensor(bits.view(torch.float32).reshape(1,1,8)),Tensor(torch.tensor([0],dtype=torch.int32)),Tensor(torch.empty(1,1,8,dtype=torch.int8)),Tensor(torch.empty(1,1,1,dtype=torch.float32))]
    assert run(*patterned) is None

    def fresh():
        return [Tensor(torch.arange(64.,dtype=torch.float32).reshape(2,2,16)),Tensor(torch.tensor([2,0],dtype=torch.int32)),Tensor(torch.full((3,2,16),-7,dtype=torch.int8)),Tensor(torch.full((3,2,2),7.,dtype=torch.float32))]
    rejected=0
    def reject(args):
        nonlocal rejected
        old_calls=len(calls)
        snapshots=[backing_bytes(t.data).clone() for t in args if isinstance(t,Tensor)]
        try: run(*args)
        except (TypeError,ValueError): pass
        else: raise AssertionError('unsupported request admitted')
        assert len(calls)==old_calls
        assert all(torch.equal(s,backing_bytes(t.data)) for s,t in zip(snapshots,[t for t in args if isinstance(t,Tensor)]))
        rejected+=1
    a=fresh();a[0]=object();reject(a)
    class WrongLayout(Tensor):
        @property
        def layout(self): return torch.sparse_coo
    a=fresh();a[0]=WrongLayout(a[0].data);reject(a)
    for i in range(4):
        a=fresh();a[i].is_cuda=False;reject(a)
        a=fresh();a[i].device=torch.device('cuda:1');reject(a)
        a=fresh();a[i]=Tensor(a[i].data.to(torch.float64));reject(a)
        a=fresh();a[i]=Tensor(a[i].data.reshape(-1,1,1,1));reject(a)
        a=fresh();a[i].is_contiguous=lambda:False;reject(a)
        a=fresh();a[i].is_neg=lambda:True;reject(a)
        a=fresh();a[i].numel=lambda:1<<63;reject(a)
    for i in (0,1,3):
        a=fresh();a[i].data_ptr=lambda:3;reject(a)
    for shape in [(2,0,16),(2,2,0),(2,2,9)]:
        a=fresh();a[0]=Tensor(torch.empty(shape));reject(a)
    a=fresh();a[0]=Tensor(torch.empty(1,2,16));reject(a)
    a=fresh();a[2]=Tensor(torch.empty(3,2,8,dtype=torch.int8));reject(a)
    a=fresh();a[3]=Tensor(torch.empty(3,2,3,dtype=torch.float32));reject(a)
    class LargeRows(Tensor):
        @property
        def shape(self): return (1<<31,2,16)
    a=fresh();a[0]=LargeRows(a[0].data);a[1].numel=lambda:1<<31;reject(a)
    a=fresh();a[0]=Tensor(torch.empty(0,1,1048584));a[1]=Tensor(torch.empty(0,dtype=torch.int32));a[2]=Tensor(torch.empty(0,1,1048584,dtype=torch.int8));a[3]=Tensor(torch.empty(0,1,131073,dtype=torch.float32));reject(a)
    a=fresh();a[0]=Tensor(torch.empty(0,65536,8));a[1]=Tensor(torch.empty(0,dtype=torch.int32));a[2]=Tensor(torch.empty(0,65536,8,dtype=torch.int8));a[3]=Tensor(torch.empty(0,65536,1,dtype=torch.float32));reject(a)
    for i in range(4):
        for j in range(i+1,4):
            a=fresh();ptr=a[i].data_ptr();a[j].data_ptr=lambda ptr=ptr:ptr;reject(a)
    for destinations in [[-1,0],[3,0],[1,1]]:
        a=fresh();a[1]=Tensor(torch.tensor(destinations,dtype=torch.int32));reject(a)
    a=fresh();a[1]=Tensor(torch.tensor([0,1,2],dtype=torch.int32)[::2]);reject(a)
    original,opid=interpreter(ORIGINAL)
    def old_run(src,dest,out,scale):
        groups=src.shape[2]
        for p in range(dest.numel()):
            for head in range(src.shape[1]):
                opid[:]=[p,head]
                original(*(Pointer(t) for t in (src,dest,out,scale)),*src.stride(),*out.stride(),*scale.stride(),groups,1<<(groups-1).bit_length(),8)
    src=torch.full((2,1,1,8),127.);dest=torch.tensor([0,1,2],dtype=torch.int32)[::2]
    out=torch.full((3,1,1,8),-7,dtype=torch.int8);scale=torch.full((3,1,1),7.)
    old_run(src,dest,out,scale)
    assert out[:,0,0,0].tolist()==[127,127,-7] and scale.flatten().tolist()==[1.,1.,7.]
    src=torch.tensor([127.,0.]*16)[::2].reshape(1,1,2,8)
    dest=torch.tensor([0],dtype=torch.int32);out=torch.full((1,1,2,8),-7,dtype=torch.int8);scale=torch.full((1,1,2),7.)
    old_run(src,dest,out,scale)
    assert out.flatten().tolist()!=[127]*16
    src=torch.full((1,1,2,8),127.);out_raw=torch.full((32,),-7,dtype=torch.int8)
    out=out_raw[::2].reshape(1,1,2,8);scale=torch.full((1,1,2),7.)
    old_run(src,dest,out,scale)
    assert bool((out_raw[1::2]!=-7).any())
    out=torch.full((1,1,2,8),-7,dtype=torch.int8);scale_raw=torch.full((4,),7.);scale=scale_raw[::2].reshape(1,1,2)
    old_run(src,dest,out,scale)
    assert scale_raw.tolist()==[1.,1.,7.,7.]
    print(f'Accepted kernel AST, {len(cases)+1} admissions, {rejected} rejections and four original stride defects passed.')
    print('Source AST and backing bytes executed on CPU; CUDA context/launch mocked; no GPU numerical claim.')


if __name__=='__main__':main()
