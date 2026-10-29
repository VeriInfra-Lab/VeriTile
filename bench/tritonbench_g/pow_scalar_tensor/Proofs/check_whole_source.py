"""Original source CPU control checks. CUDA/power/ABI primitives are adapters, not validated numerics."""
import ast
import contextlib
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'pow_scalar_tensor.py'
EXPECTED = '142398199794dc7df724c341d3f86b1af60f20a9280b695efeac5e68d757471d'

class Tensor:
    def __init__(self, data):
        self.data, self.shape, self.dtype = data, tuple(data.shape), data.dtype
        self.device = SimpleNamespace(index=0)
    def stride(self): return self.data.stride()
    def numel(self): return self.data.numel()
    def data_ptr(self): return self.data.data_ptr()

class Pointer:
    def __init__(self, tensor):
        base = tensor._base if hasattr(tensor, '_base') else tensor
        self.data = base.data
        self.offset = (tensor.data_ptr() - self.data.untyped_storage().data_ptr()) // 4
        self.type = SimpleNamespace(element_ty=torch.float32)

class BlockPointer:
    def __init__(self, base, shape, strides, offsets, block):
        self.base, self.n, self.stride, self.start, self.block = base, shape[0], strides[0], offsets[0], block[0]
        self.type = base.type
    def indices(self):
        logical = torch.arange(self.start, self.start+self.block)
        mask = logical < self.n
        return mask, self.base.offset + logical[mask]*self.stride

def storage(data):
    return torch.empty(0, dtype=torch.float32).set_(data.untyped_storage(), 0, (data.untyped_storage().nbytes()//4,), (1,))

def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    tree = ast.parse(SOURCE.read_text())
    names = {'heuristics_for_tile_size', 'heuristics_for_num_warps', 'StridedBuffer',
             'pow_func_scalar_tensor_wrapper_rank_1', 'pow_func_scalar_tensor',
             'pow_func_scalar_tensor_kernel_rank_1'}
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    for node in body:
        if isinstance(node, ast.FunctionDef): node.decorator_list = []
    current_pid, grid_size = [0], [1]
    def load(p, boundary_check):
        assert boundary_check == (0,)
        mask, indices = p.indices()
        values = torch.zeros(p.block, dtype=torch.float32)
        values[mask] = storage(p.base.data)[indices]
        return values
    def store(p, values, boundary_check):
        assert boundary_check == (0,)
        mask, indices = p.indices()
        storage(p.base.data)[indices] = values[mask]
    def cdiv(a,b): return (a+b-1)//b
    def next_power(n): return 0 if n == 0 else 1 << (n-1).bit_length()
    tl = SimpleNamespace(float32=torch.float32, tensor=object, constexpr=object,
        program_id=lambda axis: current_pid[0], num_programs=lambda axis: grid_size[0], cdiv=cdiv,
        make_block_ptr=lambda base, shape, strides, offsets, block, order: BlockPointer(base, shape, strides, offsets, block),
        load=load, store=store)
    ns = {'math': math, 'Union': Union, 'torch': SimpleNamespace(Tensor=Tensor,
        cuda=SimpleNamespace(_DeviceGuard=lambda _: contextlib.nullcontext())), 'tl': tl,
        'triton': SimpleNamespace(cdiv=cdiv,next_power_of_2=next_power), '_pow': torch.pow}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])), str(SOURCE), 'exec'),ns)
    kernel = ns['pow_func_scalar_tensor_kernel_rank_1']
    calls=[]
    class Launch:
        def __getitem__(self, grid):
            def run(base, inp, out, sx, ox, sy, oy, n, tasks, **kwargs):
                calls.append((grid, sx, sy, n, tasks, kwargs.copy()))
                grid_size[0]=grid[0]
                if n>10000: return
                for pid in reversed(range(grid[0])):
                    current_pid[0]=pid
                    kernel(torch.tensor(base,dtype=torch.float32),Pointer(inp),Pointer(out),sx,ox,sy,oy,n,tasks,
                           kwargs['tiles_per_cta'],kwargs['tile_size0'],kwargs['one_tile_per_cta'])
            return run
    ns['pow_func_scalar_tensor_kernel_rank_1']=Launch()
    wrapper, Buffer = ns['pow_func_scalar_tensor_wrapper_rank_1'],ns['StridedBuffer']
    for n in (1,511,512,513,65536*512,65536*512+1):
        class Meta(Tensor):
            def __init__(self): self.shape=(n,);self.device=SimpleNamespace(index=0)
            def numel(self): return n
            def stride(self): return (1,)
            def data_ptr(self): return 0
        if n<=10000: inp,out=Tensor(torch.ones(n)),Tensor(torch.empty(n))
        else: inp,out=Meta(),Meta()
        assert wrapper(2.,inp,out0=out) is out
        grid,sx,sy,count,tasks,kw=calls[-1]
        B=min(512,next_power(n));tiles=cdiv(n,B);C=min(65536,tiles);T=cdiv(tiles,C)
        assert grid==(C,1,1) and count==tasks==n
        assert kw==dict(tiles_per_cta=T,tile_size0=B,one_tile_per_cta=T==1,num_warps=4)
    layouts=[(2,3,3,2,False),(-2,-3,12,14,False),(0,2,4,2,False),(1,1,2,16,True)]
    for sx,sy,ix,iy,shared in layouts:
        a=torch.arange(32,dtype=torch.float32)/8
        b=a if shared else torch.full((32,),-37.,dtype=torch.float32)
        inp=Buffer(Tensor(a),shape=(5,),strides=(sx,),offset=ix)
        out=Buffer(Tensor(b),shape=(5,),strides=(sy,),offset=iy)
        assert inp.unwrap().data is a and inp.size()==(5,) and inp.numel()==5 and inp.dim()==1
        assert inp.element_size()==4 and inp.data_ptr()==a.data_ptr()+4*ix
        before_a,before_b=a.clone(),b.clone()
        inputs=before_a[ix+torch.arange(5)*sx]
        assert wrapper(2.,inp,out0=out) is out
        torch.testing.assert_close(b[iy+torch.arange(5)*sy],torch.pow(torch.tensor(2.),inputs),rtol=0,atol=0)
        mask=torch.ones(32,dtype=torch.bool);mask[iy+torch.arange(5)*sy]=False
        assert torch.equal(b[mask].view(torch.uint8),before_b[mask].view(torch.uint8))
        assert torch.equal(a[ix+torch.arange(5)*sx],inputs)
        if not shared: assert torch.equal(a.view(torch.uint8),before_a.view(torch.uint8))
    # Exercise the same original dynamic-loop JIT body with small bounded controls.
    a=torch.arange(19,dtype=torch.float32)/8;b=torch.full((25,),-37.)
    grid_size[0]=2
    for pid in (1,0):
        current_pid[0]=pid
        kernel(torch.tensor(2.),Pointer(Tensor(a)),Pointer(Tensor(b)),1,0,1,0,19,19,3,4,False)
    torch.testing.assert_close(b[:19],torch.pow(torch.tensor(2.),a),rtol=0,atol=0)
    assert torch.equal(b[19:],torch.full((6,),-37.))
    for base,exponents in [(0.,[0.,1.,-1.]),(-2.,[2.,3.,.5]),(-0.,[-3.,3.,2.]),
        (float('inf'),[-1.,0.,1.]),(float('nan'),[0.,1.,float('nan')]),(16777217.,[1.])]:
        inp=Tensor(torch.tensor(exponents,dtype=torch.float32));out=Tensor(torch.empty(len(exponents)))
        assert wrapper(base,inp,out0=out) is out
        torch.testing.assert_close(out.data,torch.pow(torch.tensor(base,dtype=torch.float32),inp.data),
                                   rtol=0,atol=0,equal_nan=True)
    calls.clear()
    try: wrapper(2.,Tensor(torch.empty(0)),out0=Tensor(torch.empty(0)))
    except ZeroDivisionError: pass
    else: raise AssertionError('original empty failure was lost')
    assert not calls
    try: wrapper(2.,Tensor(torch.ones(1)),out0=Tensor(torch.empty(2)))
    except AssertionError as error: assert 'operand shapes mismatch' in str(error)
    else: raise AssertionError('original shape assertion was lost')
    assert not calls
    print('Original SHA unchanged. Six host partitions, four signed/broadcast/shared layouts, dynamic-loop control, six power/ABI edge groups and two original failures passed.')
    print('Source AST CPU adapters only; GPU execution not run. Unchanged power/ABI primitives are model boundaries, not numerical experiment obligations.')

if __name__=='__main__': main()
