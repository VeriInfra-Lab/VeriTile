"""CPU source/host/dtype regression checks; no GPU instruction claim."""
import ast
import math
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "triton_argmax_repaired.py"


def power(n):
    if not n:
        return 0
    n -= 1
    for shift in (1, 2, 4, 8, 16, 32):
        n |= n >> shift
    return n + 1


def raw(tensor, dtype=None):
    dtype = dtype or tensor.dtype
    width = torch.empty((), dtype=dtype).element_size()
    return torch.empty(0, dtype=dtype).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes() // width,), (1,))


class Tensor:
    def __init__(self, data, *, cuda=True, pointer=None, layout=None):
        self.data, self.is_cuda, self.device = data, cuda, torch.device("cuda:0")
        self.dtype, self.shape, self.ndim = data.dtype, data.shape, data.ndim
        self.layout = data.layout if layout is None else layout
        self.pointer, self.copies = pointer, []
    def numel(self): return self.data.numel()
    def dim(self): return self.ndim
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self): return self.data.data_ptr() if self.pointer is None else self.pointer
    def untyped_storage(self): return self.data.untyped_storage()
    def element_size(self): return self.data.element_size()
    def contiguous(self):
        result = self.data.contiguous()
        wrapped = self if result is self.data else Tensor(result)
        self.copies.append(wrapped)
        return wrapped


class ElementType:
    def __init__(self, dtype): self.dtype = dtype
    def is_int(self): return not self.dtype.is_floating_point
    def is_int1(self): return self.dtype == torch.bool
    def is_int_signed(self): return self.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
    @property
    def primitive_bitwidth(self): return 8 * torch.empty((), dtype=self.dtype).element_size()


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, torch.as_tensor(offset, dtype=torch.int64)
        self.dtype = SimpleNamespace(element_ty=ElementType(tensor.dtype))
    def __add__(self, offset): return Pointer(self.tensor, self.offset + offset)


def reduce_pairs(values, axis):
    """One explicit balanced comparison tree, retaining unordered-node behavior."""
    moved = values.movedim(axis, -1)
    shape, width = moved.shape[:-1], moved.shape[-1]
    flat = moved.reshape(-1, width)
    winners, indices = [], []
    for row in flat.tolist():
        pairs = list(zip(row, range(width)))
        while len(pairs) > 1:
            pairs = [a if a[0] > b[0] or (a[0] == b[0] and a[1] < b[1]) else b
                     for a, b in zip(pairs[::2], pairs[1::2])]
        winners.append(pairs[0][0]); indices.append(pairs[0][1])
    return (torch.tensor(winners, dtype=values.dtype).reshape(shape),
            torch.tensor(indices, dtype=torch.int64).reshape(shape))


class ElementwiseAnd(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            result = node.values[0]
            for value in node.values[1:]:
                result = ast.BinOp(result, ast.BitAnd(), value)
            return ast.copy_location(result, node)
        return node


class Kernel:
    def __init__(self, node, kind):
        self.kind, self.pid, self.events = kind, 0, []
        def load(p, mask=None, other=None):
            source = raw(p.tensor)
            index = p.tensor.storage_offset() + p.offset
            if mask is None:
                return source[index]
            mask = torch.broadcast_to(torch.as_tensor(mask), index.shape)
            if p.tensor.dtype in (torch.uint16, torch.uint32, torch.uint64):
                source_values = source.tolist()
                values = [source_values[i] if live else other
                          for i,live in zip(index.reshape(-1).tolist(),mask.reshape(-1).tolist())]
                return torch.tensor(values,dtype=p.tensor.dtype).reshape(index.shape)
            result = torch.full(index.shape, other, dtype=p.tensor.dtype)
            result[mask] = source[index[mask]]
            return result
        def store(p, value, mask=None):
            index = p.tensor.storage_offset() + p.offset
            if p.tensor.dtype == torch.bool:
                target = raw(p.tensor, torch.uint8)
                value = value.to(torch.int8).view(torch.uint8)
            else:
                target = raw(p.tensor)
                value = value.to(p.tensor.dtype)
            if mask is None: target[index] = value
            else: target[index[mask]] = value[mask]
        tl = SimpleNamespace(
            program_id=lambda _: torch.tensor(self.pid, dtype=torch.int32),
            int64=torch.int64, int8=torch.int8, float32=torch.float32,
            arange=lambda a,b: torch.arange(a,b,dtype=torch.int32),
            load=load, store=store,
            max=lambda x,axis,**_: reduce_pairs(x,axis),
            argmax=lambda x,axis: reduce_pairs(x,axis)[1],
            full=lambda shape,value,dtype: torch.full(shape,value,dtype=dtype),
            where=torch.where, minimum=torch.minimum, maximum=torch.maximum)
        node.decorator_list = []
        for arg in node.args.args: arg.annotation = None
        node = ElementwiseAnd().visit(node)
        env = {"tl": tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(SOURCE), "exec"), env)
        self.body = env[node.name]
    def __getitem__(self, grid):
        def launch(*args, **meta):
            self.events.append((grid,args,meta))
            # Stage one has three Tensor pointers, followed by scalars.
            count = 3 if self.kind in (1,2) else 2
            packed = tuple(Pointer(t.data) if i < count else t for i,t in enumerate(args))
            meta.pop("num_ctas",None)
            for pid in range(grid[0]):
                self.pid = pid
                self.body(*packed, **meta)
        return launch


def harness():
    nodes = ast.parse(SOURCE.read_text()).body
    kernels = {}
    for kind,name in enumerate(("argmax_kernel_1_repaired","argmax_kernel_2_repaired","argmax_kernel_repaired"),1):
        node = next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
        kernels[name] = Kernel(node,kind)
    allocated = []
    def allocate(shape,dtype,device,zeros=False):
        out = Tensor((torch.zeros if zeros else torch.empty)(shape,dtype=dtype))
        allocated.append(out)
        return out
    fake = SimpleNamespace(**{n:getattr(torch,n) for n in (
        "float16","bfloat16","float32","float64","int8","int16","int32","int64",
        "uint8","uint16","uint32","uint64","bool","strided")})
    fake.Tensor=Tensor
    fake.empty=lambda shape,dtype,device:allocate(shape,dtype,device)
    fake.zeros=lambda shape,dtype,device:allocate(shape,dtype,device,True)
    fake.squeeze=lambda t,d:Tensor(t.data.squeeze(d))
    fake.cuda=SimpleNamespace(device=lambda _:nullcontext())
    selected = [n for n in nodes if isinstance(n,(ast.Assign,ast.FunctionDef))
                and not (isinstance(n,ast.FunctionDef) and n.name in kernels)]
    for node in selected:
        if isinstance(node,ast.FunctionDef): node.decorator_list=[]
    env={"torch":fake,"triton":SimpleNamespace(next_power_of_2=power,cdiv=lambda a,b:(a+b-1)//b),
         "math":math,**kernels}
    exec(compile(ast.fix_missing_locations(ast.Module(selected,[])),str(SOURCE),"exec"),env)
    return env, kernels, allocated


def staged_expected(values, dtype):
    size=values.numel(); block=power(math.ceil(math.sqrt(size)))
    count=(size+block-1)//block
    maximum,indices=[],[]
    for start in range(0,size,block):
        chunk=values[start:start+block]
        maximum.append(chunk.max().item())
        indices.append(start+chunk.argmax().item())
    mid=torch.tensor(maximum,dtype=torch.float32)
    if not dtype.is_floating_point:
        signed=dtype in (torch.int8,torch.int16,torch.int32,torch.int64,torch.bool)
        bits=8*torch.empty((),dtype=dtype).element_size()
        exponent=bits-1 if signed else bits
        lower=-(1<<exponent) if signed else 0
        upper=(1<<exponent)-(1<<max(exponent-24,0))
        mid=torch.nan_to_num(mid,nan=0).clamp(lower,upper)
        mid=mid.to(torch.int8).ne(0) if dtype==torch.bool else mid.to(dtype)
    else: mid=mid.to(dtype)
    # Independent scalar comparison over stored intermediate values.
    decoded=mid.tolist();winner=max(range(count),key=lambda i:(decoded[i],-i))
    return indices[winner]


def checks():
    env,kernels,allocated=harness(); call=env["argmax"]; admitted=0
    for dtype in env["_MID_DTYPES"]:
        base=torch.full((43,),-99.);base[3:36]=torch.linspace(-3,8,33)
        x=Tensor(base[3:36]);saved=bytes(base.untyped_storage())
        out=call(x,dtype=dtype,keepdim=True)
        assert out.data.item()==staged_expected(x.data,dtype)
        assert out.shape==(1,) and out.dtype==torch.int64 and not x.copies
        assert out is allocated[-1]
        assert bytes(base.untyped_storage())==saved
        admitted+=1
    # Intermediate rounding changes the answer; preserve dtype semantics.
    x=Tensor(torch.tensor([1.0001]+[-9.]*3+[1.0002]+[-9.]*11))
    assert call(x,dtype=torch.float16).data.item()==0 and torch.argmax(x.data).item()==4
    admitted+=1
    # Fractional bool conversion truncates through i8 before testing the byte.
    x=Tensor(torch.tensor([0.2]+[0.1]*3+[0.8]+[0.1]*11))
    assert call(x,dtype=torch.bool).data.item()==0
    admitted+=1
    # Negative i8 bytes load as true even though fractional positives truncate to zero.
    x=Tensor(torch.tensor([0.2]+[-9.]*3+[0.8]+[-9.]*11))
    assert call(x,dtype=torch.bool).data.item()==8
    admitted+=1
    # Floating unordered comparisons may select second-stage padding; index zero is initialized.
    x=Tensor(torch.full((24,),float("nan")))
    assert call(x,dtype=torch.float32).data.item()==0
    admitted+=1
    for shape,dim,keep,view in [
        ((2,131,3),1,False,False),((2,131,3),-2,True,False),
        ((131,2),0,False,True),((2,3,131),-1,True,False)]:
        value=torch.randn(shape)
        if view:value=value.T
        x=Tensor(value);saved=bytes(value.untyped_storage())
        out=call(x,dim=dim,keepdim=keep,dtype=object())
        torch.testing.assert_close(out.data,torch.argmax(value,dim=dim,keepdim=keep))
        assert out.data.untyped_storage().data_ptr()==allocated[-1].data.untyped_storage().data_ptr()
        assert (out is allocated[-1])==keep
        assert bytes(value.untyped_storage())==saved and len(x.copies)==1
        admitted+=1
    x=Tensor(torch.ones(2,260,3));x.data[:,130,:]=9;x.data[:,259,:]=9
    assert torch.equal(call(x,dim=1).data,torch.full((2,3),130,dtype=torch.int64))
    admitted+=1
    x=Tensor(torch.arange(131.).view(1,131).expand(3,131))
    assert torch.equal(call(x,dim=1).data,torch.full((3,),130,dtype=torch.int64))
    admitted+=1
    for shape,axis in (((0,3,4),1),((2,3,0),1)):
        x=Tensor(torch.empty(shape));prior=sum(len(k.events) for k in kernels.values())
        out=call(x,dim=axis,dtype=object())
        assert out.numel()==0 and sum(len(k.events) for k in kernels.values())==prior
        admitted+=1
    x=Tensor(torch._neg_view(torch.empty(0,3,4)))
    assert call(x,dim=1).shape==(0,4)
    admitted+=1
    for keep in (False,True):
        x=Tensor(torch.tensor(7.))
        out=call(x,keepdim=keep)
        assert out.shape==() and out.data.item()==0 and out is allocated[-1]
        admitted+=1
    controls=[
        (None,{}),(Tensor(torch.ones(4),cuda=False),{}),
        (Tensor(torch.ones(4),layout=object()),{}),
        (Tensor(torch.ones(4),pointer=(1<<64)-4),{}),
        (Tensor(torch.tensor(7.)),{"dim":0}),
        (Tensor(torch.ones(4,dtype=torch.float16)),{}),
        (Tensor(torch.ones(2,3).T),{}),(Tensor(torch.ones(4),pointer=4097),{}),
        (Tensor(torch.empty(0)),{}),(Tensor(torch.empty(2,0,3)),{"dim":1}),
        (Tensor(torch.ones(2,3)),{"dim":2}),(Tensor(torch.ones(2,3)),{"dim":-3}),
        (Tensor(torch.ones(2,3)),{"dim":True}),(Tensor(torch.ones(4)),{"keepdim":1}),
        (Tensor(torch.ones(4)),{"dtype":object()}),
        (Tensor(torch._neg_view(torch.ones(4))),{}),
        (Tensor(torch._neg_view(torch.ones(2,3))),{"dim":1})]
    for x,kwargs in controls:
        prior=sum(len(k.events) for k in kernels.values())
        saved=bytes(x.data.untyped_storage()) if isinstance(x,Tensor) else None
        try:call(x,**kwargs)
        except (TypeError,ValueError,IndexError,AssertionError):pass
        else:raise AssertionError(("admitted rejection control",kwargs))
        assert sum(len(k.events) for k in kernels.values())==prior
        if saved is not None: assert bytes(x.data.untyped_storage())==saved
    geometry_controls=[
        (env["_check_global_geometry"],(1,0,1,1)),
        (env["_check_global_geometry"],(1,1<<21,1,1)),
        (env["_check_global_geometry"],(1,1,2,1)),
        (env["_check_global_geometry"],((1<<63)-1,1,1,1)),
        (env["_check_dimension_geometry"],(1,(1<<63)-128,1)),
        (env["_check_dimension_geometry"],((1<<60),1,1)),
        (env["_check_fresh"],(Tensor(torch.ones(4)),)*2)]
    for guard,args in geometry_controls:
        try: guard(*args)
        except ValueError: pass
        else: raise AssertionError(("admitted invalid geometry/storage",args))
    env["_check_dimension_geometry"](0,1<<90,1)
    env["_check_dimension_geometry"](1,1<<90,0)
    for total in (0,1,65535,65536,131071):
        segments=list(env["_launch_segments"](total))
        assert [i for start,grid in segments for i in range(start,start+grid[0])]==list(range(total))
        assert all(0<grid[0]<=65535 for _,grid in segments)
    text=SOURCE.read_text()
    hashes={
        "argmax_kernel_1_repaired":"02e20deb1dbd20fc69c64be7a95e19f20edb275b48ad44bb1124b21fa41588ed",
        "argmax_kernel_2_repaired":"7c3ffb73548ecd995186c6667e2aa6391b700d590e764d24868eb6be5d3064d0",
        "argmax_kernel_repaired":"284d53fd882833bf23d240fa794a9f2e244fc8c0cfcc1201fcc9fe765ff2d60e"}
    for node in ast.parse(text).body:
        if isinstance(node,ast.FunctionDef) and node.name in hashes:
            assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==hashes[node.name]
    assert "_fresh_fp32_logical" not in text and "_gather_fp32_bytes" not in text
    assert ".Proofs." not in (HERE/"Kernels.lean").read_text()
    assert "FPEquiv" not in (HERE/"Correct.lean").read_text().split("namespace")[0]
    assert "Correct" not in (HERE/"FPEquiv.lean").read_text().split("namespace")[0]
    print(f"{admitted} source/API/dtype/axis/empty controls and {len(controls)+len(geometry_controls)} rejections passed; caller storage preserved")


if __name__=="__main__":
    checks()
