"""Execute the actual JIT AST on CPU storage; CUDA device and launch are mocked."""
import ast
import contextlib
import importlib.util
import hashlib
import sys
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
SOURCE = HERE / 'quantize_copy_kv_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/quantize_copy_kv/quantize_copy_kv.py'
ACCEPTED_AST_SHA256 = '204c30e21c4208b27c96c6788904cb7fdc3257d6e5709a213bd12696a10a87b7'
NAME = '_fwd_kernel_destindex_copy_quantize_kv'
spec = importlib.util.spec_from_file_location('copy_test_helpers', HERE.parent / 'destindex_copy_kv1/check_contract.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
function, raw, backing_bytes, Pointer = helpers.function, helpers.raw, helpers.backing_bytes, helpers.Pointer


class Tensor(helpers.Tensor):
    def element_size(self): return self.data.element_size()


def source_binding():
    assert [(p["mode"], p["statements"]) for p in check_manifest(ROOT, HERE / "jit-correspondence.json")] == [("pasted_body", 14), ("pasted_body", 14)]
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
    pid = [0]
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
    ns = {'tl': SimpleNamespace(program_id=lambda _: torch.tensor(pid[0], dtype=torch.int32),
        arange=lambda a,b: torch.arange(a,b,dtype=torch.int32),load=load,store=store,
        int64=torch.int64,int8=torch.int8,float16=torch.float16,float32=torch.float32,
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
                    pid[0]=p
                    kernel(*(Pointer(t.data) for t in (k,dest,out,scale)),*args,kw['BLOCK_DMODEL'],kw['BLOCK_HEAD'])
            return run
    module._fwd_kernel_destindex_copy_quantize_kv=Launch()
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        int64=torch.int64,int8=torch.int8,float16=torch.float16,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    run=module.destindex_copy_quantize_kv
    cases=[(0,3,4),(1,1,1),(2,3,2),(3,3,8),(2,7,16),(2,12,64),(1,5,256),(2,3,4)]
    for count,heads,dims in cases:
        size=(count+1)*heads*dims
        src=torch.arange(size+6,dtype=torch.float32)[1:1+size].reshape(count+1,heads,dims)
        src.copy_((src%255)-127)
        src[:,:,-1]=127
        src[::2]*=2
        if count: src[0,0]=0
        output=torch.full(((count+3)*heads*dims+10,),-7,dtype=torch.int8)[3:3+(count+3)*heads*dims].reshape(count+3,heads,dims)
        scale=torch.full(((count+3)*heads+6,),7.,dtype=torch.float16)[2:2+(count+3)*heads].reshape(count+3,heads,1)
        dest=torch.arange(count+2,dtype=torch.int64)[1:count+1].flip(0)
        args=[Tensor(t) for t in (src,dest,output,scale)]
        snapshots=[backing_bytes(t).clone() for t in (src,dest,output,scale)]
        expected,expected_scale=output.clone(),scale.clone()
        for s,d in enumerate(dest.tolist()):
            maximum=src[s].abs().amax(1,keepdim=True)
            row_scale=(maximum/127).half()
            expected[d]=(src[s]/torch.where(maximum==0,1.,row_scale.float())).to(torch.int8)
            expected_scale[d]=row_scale
        old_calls=len(calls)
        assert run(*args) is None
        assert torch.equal(output,expected) and torch.equal(scale,expected_scale)
        for t,snapshot in zip((src,dest,output,scale),snapshots):
            if t is src or t is dest: assert torch.equal(backing_bytes(t),snapshot)
            start=t.storage_offset()*t.element_size();end=start+t.numel()*t.element_size()
            assert torch.equal(backing_bytes(t)[:start],snapshot[:start]) and torch.equal(backing_bytes(t)[end:],snapshot[end:])
        assert len(calls)-old_calls==int(count>0)
        if count: assert all(a is b for a,b in zip(calls[-1][1:5],args))
    # The guard admits every FP32 pattern. This checks admission and byte framing,
    # not a GPU-defined value for NaN, infinity or out-of-range integer casts.
    bits=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1,1065353216,-1082130432],dtype=torch.int32)
    patterned=[Tensor(bits.view(torch.float32).reshape(1,1,8)),Tensor(torch.tensor([0],dtype=torch.int64)),Tensor(torch.empty(1,1,8,dtype=torch.int8)),Tensor(torch.empty(1,1,1,dtype=torch.float16))]
    assert run(*patterned) is None

    def fresh():
        return [Tensor(torch.arange(16.,dtype=torch.float32).reshape(2,2,4)),Tensor(torch.tensor([2,0],dtype=torch.int64)),Tensor(torch.full((3,2,4),-7,dtype=torch.int8)),Tensor(torch.full((3,2,1),7.,dtype=torch.float16))]
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
    for shape in [(2,0,4),(2,2,0),(2,2,3)]:
        a=fresh();a[0]=Tensor(torch.empty(shape));reject(a)
    a=fresh();a[0]=Tensor(torch.empty(1,2,4));reject(a)
    a=fresh();a[2]=Tensor(torch.empty(3,2,8,dtype=torch.int8));reject(a)
    a=fresh();a[3]=Tensor(torch.empty(3,2,2,dtype=torch.float16));reject(a)
    class LargeRows(Tensor):
        @property
        def shape(self): return (1<<31,2,4)
    a=fresh();a[0]=LargeRows(a[0].data);a[1].numel=lambda:1<<31;reject(a)
    a=fresh();a[0]=Tensor(torch.empty(0,1025,1024));a[1]=Tensor(torch.empty(0,dtype=torch.int64));a[2]=Tensor(torch.empty(0,1025,1024,dtype=torch.int8));a[3]=Tensor(torch.empty(0,1025,1,dtype=torch.float16));reject(a)
    for i in range(4):
        for j in range(i+1,4):
            a=fresh();ptr=a[i].data_ptr();a[j].data_ptr=lambda ptr=ptr:ptr;reject(a)
    for destinations in [[-1,0],[3,0],[1,1]]:
        a=fresh();a[1]=Tensor(torch.tensor(destinations,dtype=torch.int64));reject(a)
    a=fresh();a[1]=Tensor(torch.tensor([0,1,2],dtype=torch.int64)[::2]);reject(a)
    original,opid=interpreter(ORIGINAL)
    src=torch.full((2,1,1),127.);dest=torch.tensor([0,1,2],dtype=torch.int64)[::2]
    out=torch.full((3,1,1),-7,dtype=torch.int8);scale=torch.full((3,1,1),7.,dtype=torch.float16)
    for p in range(2):
        opid[0]=p
        original(*(Pointer(t) for t in (src,dest,out,scale)),1,1,1,1,1,1,1,1,1,1,1,1)
    assert out.flatten().tolist()==[127,127,-7] and scale.flatten().tolist()==[1.,1.,7.]
    print(f'Accepted kernel AST, {len(cases)+1} admissions, {rejected} rejections and original metadata-stride defect passed.')
    print('Source AST and backing bytes executed on CPU; CUDA context/launch mocked; no GPU numerical claim.')


if __name__=='__main__':main()
