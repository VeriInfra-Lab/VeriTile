"""Actual repaired JIT/host replay on CPU backing storage; CUDA is mocked."""
import ast
import contextlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'rotary_emb_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/rotary_emb/rotary_emb.py'
spec = importlib.util.spec_from_file_location('fixtures', HERE.parent / 'swiglu_fwd/check_contract.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)

class Tensor(f.Tensor):
    def stride(self, axis=None): return self.data.stride() if axis is None else self.data.stride(axis)
    def is_contiguous(self): return self.data.is_contiguous()
    def is_conj(self): return self.data.is_conj()
    def numel(self): return self.data.numel()


def host(path=SOURCE):
    ids, events = [0, 0], []
    def load(ptr, mask, other=0.):
        offset, mask = torch.broadcast_tensors(ptr.offset, mask)
        result = torch.full(offset.shape, other, dtype=torch.float32)
        result[mask] = f.storage(ptr.tensor)[ptr.tensor.storage_offset() + offset[mask]]
        return result
    def store(ptr, value, mask):
        offset, mask, value = torch.broadcast_tensors(ptr.offset, mask, value)
        f.storage(ptr.tensor)[ptr.tensor.storage_offset() + offset[mask]] = value[mask]
    fn = f.function(path, '_rotary_kernel')
    fn.decorator_list = []
    for arg in fn.args.args: arg.annotation = None
    ns = {'tl': SimpleNamespace(program_id=lambda axis: torch.tensor(ids[axis]),
        arange=torch.arange, int64=torch.int64, load=load, store=store)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), ns)
    kernel = ns['_rotary_kernel']
    class Launch:
        def __getitem__(self, grid):
            def run(q, k, cos, sin, *args, **kwargs):
                events.append(('launch', grid, kwargs.copy()))
                kwargs.pop('num_warps'); kwargs.pop('num_stages')
                for head in reversed(range(grid[0])):
                    ids[0] = head
                    for token in reversed(range(grid[1])):
                        ids[1] = token
                        kernel(*(f.Pointer(t.data) for t in (q,k,cos,sin)), *args, **kwargs)
            return run
    @contextlib.contextmanager
    def device(d): events.append(('device',d)); yield
    ns.update(torch=SimpleNamespace(Tensor=Tensor, strided=torch.strided,float32=torch.float32,
        no_grad=lambda:lambda fn:fn,cuda=SimpleNamespace(device=device)),math=math,
        triton=SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()),
        _rotary_kernel=Launch())
    fn = f.function(path,'rotary_emb_fwd')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])), str(path),'exec'),ns)
    return ns['rotary_emb_fwd'],events


def reference(x,c,s,r):
    result=x.clone()
    a,b=x[...,0:r:2],x[...,1:r:2]
    result[...,0:r:2]=a*c[:,None,0:r:2]-b*s[:,None,0:r:2]
    result[...,1:r:2]=a*s[:,None,1:r:2]+b*c[:,None,1:r:2]
    return result


def main():
    repaired,events=host()
    # Preserve the exact source arithmetic and default launch fusion policy.
    def outputs(path):
        return {n.targets[0].id:ast.dump(n.value) for n in ast.walk(f.function(path,'_rotary_kernel'))
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in
            ('out0','out1','out_k0','out_k1')}
    assert outputs(SOURCE)==outputs(ORIGINAL)
    assert 'enable_fp_fusion' not in SOURCE.read_text()
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    shapes=[(1,1,1,2,2,2,1.),(17,3,5,6,8,10,1.),(3,5,2,7,8,10,.9),
        (2,1,7,10,10,12,1.),(2,2,1,128,128,130,1.),(1,1,1,258,260,262,1.),
        (0,2,3,6,6,6,1.),(2,2,1,6,0,0,0.),(2,3,1,6,6,8,.5)]
    # Last configuration selects an odd rotary width and is checked separately.
    shapes.pop()
    torch.manual_seed(193)
    for t,hq,hk,d,cw,sw,factor in shapes:
        args=[]; raw=[]; before=[]
        for shape in ((t,hq,d),(t,hk,d),(t,cw),(t,sw)):
            n=math.prod(shape)
            data=torch.randint(-3,4,(n+7,),dtype=torch.int64).float()
            args.append(Tensor(data[:n].reshape(shape)));raw.append(data);before.append(data.clone())
        r=int(d*factor)
        expected=[reference(args[i].data.clone(),args[2].data,args[3].data,r) for i in (0,1)]
        events.clear();assert repaired(*args,factor) is None
        for i in (0,1):
            assert torch.equal(args[i].data,expected[i])
            assert torch.equal(raw[i][args[i].numel():],before[i][args[i].numel():])
        assert torch.equal(raw[2],before[2]) and torch.equal(raw[3],before[3])
        assert sum(e[0]=='launch' for e in events)==bool(t*r)
    base=[Tensor(torch.randn(2,2,6)),Tensor(torch.randn(2,3,6)),Tensor(torch.randn(2,6)),Tensor(torch.randn(2,8))]
    cases=[]
    def bad(i,t,error=ValueError):
        args=base.copy();args[i]=t;cases.append((args,1.,error))
    bad(0,[],TypeError);bad(0,Tensor(torch.randn(2,2,6,dtype=torch.float64)),TypeError)
    bad(0,Tensor(base[0].data,'cpu',False),TypeError)
    bad(1,Tensor(base[1].data,'cuda:0'))
    bad(0,Tensor(torch.randn(2,2,12)[...,::2]));bad(0,Tensor(torch._neg_view(base[0].data)))
    bad(0,Tensor(torch.randn(25)[1:].reshape(2,2,6)))
    bad(0,Tensor(torch.randn(2,0,6)));bad(0,Tensor(torch.randn(2,6)))
    bad(3,base[2]);bad(1,Tensor(torch.randn(1,3,6)));bad(2,Tensor(torch.randn(2,4)))
    for factor in (float('nan'),float('inf'),-.1,1.1,True, '1', .5): cases.append((base,factor,ValueError))
    for remainder in (1,2,3):
        bytes_=torch.zeros(100,dtype=torch.uint8).untyped_storage()[remainder:remainder+96]
        bad(0,Tensor(torch.empty(0).set_(bytes_,0,(2,2,6),(12,6,1))))
    # Different storage handles and start pointers can still overlap physically.
    backing=torch.arange(160,dtype=torch.uint8).untyped_storage()
    a=torch.empty(0).set_(backing[:96],0,(2,2,6),(12,6,1))
    b=torch.empty(0).set_(backing[4:148],0,(2,3,6),(18,6,1))
    args=base.copy();args[:2]=[Tensor(a),Tensor(b)];cases.append((args,1.,ValueError))
    for args,factor,error in cases:
        events.clear()
        try: repaired(*args,factor)
        except error: pass
        else: raise AssertionError('unsupported input admitted')
        assert not any(e[0]=='launch' for e in events)
    # Concrete original sine-stride failure, and missing K-head coverage.
    old,_=host(ORIGINAL)
    for t,hq,hk,d,cw,sw in ((2,1,1,4,4,6),(1,1,5,4,4,4)):
        args=[Tensor(torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)+1)
            for shape in ((t,hq,d),(t,hk,d),(t,cw),(t,sw))]
        expected=reference(args[1].data.clone(),args[2].data,args[3].data,d)
        old(*args)
        assert not torch.equal(args[1].data,expected)
    print(f'{len(shapes)} complete CPU source/storage paths; {len(cases)} prewrite rejection paths passed.')
    print('Original sine-stride and K-head defects reproduced. CUDA and numeric conformance are not tested.')

if __name__=='__main__':main()
