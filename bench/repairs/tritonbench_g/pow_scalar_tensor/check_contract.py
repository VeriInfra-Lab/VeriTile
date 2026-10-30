"""Original source CPU control checks. CUDA/power/ABI primitives are adapters, not validated numerics."""
import ast
import contextlib
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import torch

SOURCE = Path(__file__).resolve().parent / 'pow_scalar_tensor_repaired.py'
ORIGINAL = Path(__file__).resolve().parents[3] / 'tritonbench_g/pow_scalar_tensor/pow_scalar_tensor.py'
EXPECTED = '142398199794dc7df724c341d3f86b1af60f20a9280b695efeac5e68d757471d'

class Tensor:
    def __init__(self, data):
        self.data, self.shape, self.dtype = data, tuple(data.shape), data.dtype
        self.device = torch.device('cuda:0')
        self.layout, self.is_cuda = data.layout, True
    def stride(self): return self.data.stride()
    def numel(self): return self.data.numel()
    def data_ptr(self): return self.data.data_ptr()
    def is_neg(self): return self.data.is_neg()
    def untyped_storage(self): return self.data.untyped_storage()

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
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == EXPECTED
    orig_tree=ast.parse(ORIGINAL.read_text())
    new_tree=ast.parse(SOURCE.read_text())
    for name in ('StridedBuffer','pow_func_scalar_tensor','pow_func_scalar_tensor_kernel_rank_1'):
        a=next(n for n in orig_tree.body if getattr(n,'name',None)==name)
        b=next(n for n in new_tree.body if getattr(n,'name',None)==name)
        assert ast.dump(a)==ast.dump(b),name
    guard_functions={n.name:n for n in new_tree.body if isinstance(n,ast.FunctionDef)}
    assert sum(isinstance(n,ast.Raise) for n in ast.walk(guard_functions['_checked_view']))==12
    assert sum(isinstance(n,ast.Raise) for n in ast.walk(guard_functions['_check_inputs']))==7
    forbidden={'clone','contiguous','reshape','flatten','copy_','resolve_neg'}
    assert not any(isinstance(n,ast.Attribute) and n.attr in forbidden for key in ('_checked_view','_check_inputs')
                   for n in ast.walk(guard_functions[key]))
    tree = ast.parse(SOURCE.read_text())
    names = {'_checked_view','_check_inputs','heuristics_for_tile_size', 'heuristics_for_num_warps', 'StridedBuffer',
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
    ns = {'math': math, 'Union': Union, 'torch': SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
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
            def __init__(self, address):
                self.shape=(n,);self.device=torch.device('cuda:0');self.address=address
                self.dtype=torch.float32;self.layout=torch.strided;self.is_cuda=True
            def numel(self): return n
            def stride(self): return (1,)
            def data_ptr(self): return self.address
            def is_neg(self): return False
            def untyped_storage(self): return SimpleNamespace(data_ptr=lambda:self.address,nbytes=lambda:4*n)
        if n<=10000: inp,out=Tensor(torch.ones(n)),Tensor(torch.empty(n))
        else: inp,out=Meta(4096),Meta(1<<48)
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
    # Distinct overlapping backing storage objects are admitted when the actual
    # accessed spans are disjoint. DLPack shares memory without a common base.
    a=torch.arange(32,dtype=torch.float32)/8
    inp=Tensor(a[2:7]);out=Tensor(torch.from_dlpack(a[16:21]))
    xb,xbytes=inp.untyped_storage().data_ptr(),inp.untyped_storage().nbytes()
    yb,ybytes=out.untyped_storage().data_ptr(),out.untyped_storage().nbytes()
    assert xb != yb and max(xb,yb)<min(xb+xbytes,yb+ybytes)
    assert inp.data_ptr()+4*inp.numel()<=out.data_ptr()
    before=a.clone()
    assert wrapper(2.,inp,out0=out) is out
    torch.testing.assert_close(a[16:21],torch.pow(torch.tensor(2.),before[2:7]),rtol=0,atol=0)
    exterior=torch.ones(32,dtype=torch.bool);exterior[16:21]=False
    assert torch.equal(a[exterior].view(torch.uint8),before[exterior].view(torch.uint8))
    print('DLPack overlapping-backing/disjoint-active output and physical exterior frame passed.')
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
    except ValueError as error: assert 'operand shapes mismatch' in str(error)
    else: raise AssertionError('original shape assertion was lost')
    assert not calls
    bad=[]
    valid=Tensor(torch.ones(4)); other=Tensor(torch.empty(4))
    bad += [(2,valid,other,TypeError), (2.,[],other,TypeError),
        (2.,Tensor(torch.ones(4,dtype=torch.float64)),other,TypeError),
        (2.,Tensor(torch.ones(2,2)),Tensor(torch.empty(2,2)),ValueError),
        (2.,Tensor(torch._neg_view(torch.ones(4))),other,ValueError),
        (2.,valid,Buffer(other,shape=(4,),strides=(0,)),ValueError),
        (2.,valid,valid,ValueError),
        (2.,Buffer(valid,shape=(4,),strides=(1,),offset=1),other,ValueError),
        (2.,Buffer(valid,shape=(4,),strides=(-1,),offset=0),other,ValueError),
        (2.,Buffer(valid,shape=(1,),strides=(1<<63,)),Tensor(torch.empty(1)),ValueError),
        (2.,Buffer(valid,shape=(1,),strides=(-(1<<63)-1,)),Tensor(torch.empty(1)),ValueError),
        (2.,Buffer(valid,shape=(4,),strides=(1.0,)),other,TypeError)]
    unaligned=Buffer(valid,shape=(4,),strides=(1,));unaligned._data_ptr += 1
    foreign=object.__new__(Buffer);foreign._base=[]
    bad += [(2.,Tensor(torch.eye(2).to_sparse()),other,TypeError),
        (2.,Buffer(valid,shape=(4,),strides=()),other,ValueError),
        (2.,Buffer(valid,shape=(4.,),strides=(1,)),other,TypeError),
        (2.,unaligned,other,ValueError),(2.,foreign,other,TypeError)]
    class BadAllocation(Tensor):
        def untyped_storage(self): return SimpleNamespace(data_ptr=lambda:1<<64,nbytes=lambda:16)
        def data_ptr(self): return 1<<64
    bad.append((2.,BadAllocation(torch.ones(4)),other,ValueError))
    cpu=Tensor(torch.ones(4));cpu.is_cuda=False
    peer=Tensor(torch.empty(4));peer.device=torch.device('cuda:1')
    bad += [(2.,cpu,other,ValueError),(2.,valid,peer,ValueError)]
    for scalar,inp,out,exception in bad:
        calls.clear()
        try: wrapper(scalar,inp,out0=out)
        except exception: pass
        else: raise AssertionError('out-of-contract argument accepted')
        assert not calls
    n=1<<31
    calls.clear()
    try: wrapper(2.,Meta(4096),out0=Meta(1<<48))
    except ValueError as error: assert '2**31' in str(error)
    else: raise AssertionError('oversized index domain admitted')
    assert not calls
    print(f'{len(bad)+1} checked-domain rejection cases passed.')
    print('Original SHA unchanged. Six host partitions, four signed/broadcast/shared layouts, dynamic-loop control, six power/ABI edge groups and two original failures passed.')
    print('Source AST CPU adapters only; GPU execution not run. Unchanged power/ABI primitives are model boundaries, not numerical experiment obligations.')

if __name__=='__main__': main()
