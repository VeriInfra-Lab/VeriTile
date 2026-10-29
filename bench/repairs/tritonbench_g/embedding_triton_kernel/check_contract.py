"""Targeted CPU source replay; CUDA context and launch are mocked, not validated."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / "tritonbench_g/embedding_triton_kernel/embedding_triton_kernel.py"


class Tensor:
    def __init__(self, value, **overrides):
        self.value = value
        self.__dict__.update(overrides)
    def __getattr__(self, name):
        if name == "is_cuda":
            return True
        if name == "device":
            return torch.device("cuda:0")
        return getattr(self.value, name)


class Tile(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)
    def to(self, dtype):
        return Tile(self, dtype)


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset
    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)
    __radd__ = __add__
    def storage(self):
        t = self.tensor.value
        storage = t.untyped_storage()
        raw = (ctypes.c_ubyte * storage.nbytes()).from_address(storage.data_ptr())
        dtype = np.int32 if t.dtype == torch.int32 else np.float32
        array = np.frombuffer(raw, dtype=dtype)
        return array, (t.data_ptr() - storage.data_ptr()) // 4


class Language:
    int64, int32, constexpr = np.int64, np.int32, object()
    pid = 0
    @classmethod
    def program_id(cls, axis):
        assert axis == 0
        return Tile(cls.pid, np.int32)
    @staticmethod
    def arange(first, last):
        return Tile(np.arange(first, last, dtype=np.int32))
    @staticmethod
    def multiple_of(value, alignment):
        return value
    @staticmethod
    def where(mask, yes, no):
        return Tile(np.where(mask, yes, no))
    @staticmethod
    def load(pointer, mask=True, other=0):
        array, base = pointer.storage()
        offsets, masks = np.broadcast_arrays(pointer.offset, mask)
        values = np.empty(offsets.shape, dtype=array.dtype)
        # Explicitly model tl.load's conversion of an int32 default.
        if array.dtype == np.int32:
            other = ((int(other) + 2**31) % 2**32) - 2**31
        values[...] = other
        for index in np.ndindex(offsets.shape):
            if masks[index]:
                address = base + int(offsets[index])
                if not 0 <= address < len(array):
                    raise IndexError("active source load outside allocation")
                values[index] = array[address]
        return Tile(values)
    @staticmethod
    def store(pointer, values, mask=True):
        array, base = pointer.storage()
        offsets, values, masks = np.broadcast_arrays(pointer.offset, values, mask)
        for index in np.ndindex(offsets.shape):
            if masks[index]:
                address = base + int(offsets[index])
                if not 0 <= address < len(array):
                    raise IndexError("active source store outside allocation")
                array[address] = values[index]


class Kernel:
    launches = 0
    def __init__(self, function):
        self.function = function
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            Kernel.launches += 1
            kwargs.pop("num_warps", None)
            kwargs.pop("num_stages", None)
            args = [Pointer(x) if isinstance(x, Tensor) else x for x in args]
            for pid in range(grid[0]):
                Language.pid = pid
                self.function(*args, **kwargs)
        return launch


def load_source(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and not node.name.startswith("test_")]
    env = {"torch": SimpleNamespace(Tensor=Tensor, float32=torch.float32, int32=torch.int32,
            strided=torch.strided, no_grad=lambda: lambda f: f,
            cuda=SimpleNamespace(device=lambda _: nullcontext())),
           "triton": SimpleNamespace(jit=Kernel, cdiv=lambda n,b:(n+b-1)//b,
            next_power_of_2=lambda n: 1 << max(0,n-1).bit_length()), "tl": Language}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
    return env


def call_case(ids, weight, first, last):
    call_case.count += 1
    output = torch.full((len(ids), weight.shape[1]), float("nan"), dtype=torch.float32)
    expected = torch.zeros_like(output)
    for r, token in enumerate(ids.tolist()):
        if weight.shape[1] and first <= token < last:
            expected[r] = weight[token-first]
    before_weight, before_ids = weight.clone(), ids.clone()
    launches = Kernel.launches
    result = repaired(Tensor(ids), Tensor(weight), first, last, Tensor(output))
    assert result is None
    assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(weight.view(torch.int32), before_weight.view(torch.int32))
    assert torch.equal(ids, before_ids)
    assert Kernel.launches-launches == int(len(ids)>0 and weight.shape[1]>0)


def reject(*args):
    launches = Kernel.launches
    try:
        repaired(*args)
    except (ValueError, TypeError):
        reject.count += 1
        assert launches == Kernel.launches
    else:
        raise AssertionError("unsupported input admitted")


call_case.count = 0
reject.count = 0
repaired_env = load_source(HERE / "embedding_triton_kernel_repaired.py")
repaired = repaired_env["embedding_repaired"]
original = load_source(ORIGINAL)["embedding"]
# Signed zeros, infinities, subnormal and NaN payloads are compared as words.
words = torch.tensor([0, -2147483648, 1, 2139095040, 2143289345, -1], dtype=torch.int32)
weight = words.view(torch.float32).reshape(2,3)
for ids in ([], [-4], [-3,-2,-1,0,1], list(range(-4,65))):
    call_case(torch.tensor(ids,dtype=torch.int32), weight, -3,-1)
call_case(torch.tensor([2147483647,-2147483648],dtype=torch.int32), weight[:1], 2147483647,2**31)
call_case(torch.tensor([-2147483648,0],dtype=torch.int32), weight[:1], -2**31,-2**31+1)
for first,last in ((10,10),(11,-11),(2**100,2**101),(-2**101,-2**100)):
    call_case(torch.tensor([-2,0,2],dtype=torch.int32), weight[:0],first,last)
call_case(torch.tensor([1,2],dtype=torch.int32), torch.empty(0,0),-2**100,2**100)
# Contiguous offset inputs and output keep their own actual pointers.
ids = Tensor(torch.tensor([99,0,1,99],dtype=torch.int32)[1:3])
w = Tensor(torch.arange(12,dtype=torch.float32).reshape(4,3)[1:3])
o = Tensor(torch.full((4,3),-1.,dtype=torch.float32)[1:3])
assert repaired(ids,w,0,2,o) is None
assert torch.equal(o.value,w.value)
base = (Tensor(torch.tensor([0,1],dtype=torch.int32)), w, 0, 2, Tensor(torch.zeros(2,3)))
reject(None,*base[1:])
reject(base[0],w,False,2,base[4])
reject(Tensor(base[0].value.to(torch.int64)),*base[1:])
reject(base[0],Tensor(w.value.to(torch.float64)),0,2,base[4])
reject(Tensor(base[0].value,is_cuda=False),*base[1:])
reject(base[0],Tensor(w.value,device=torch.device('cuda:1')),0,2,base[4])
reject(Tensor(torch.tensor([[0,1]],dtype=torch.int32)),*base[1:])
reject(base[0],w,0,2,Tensor(torch.zeros(3,3)))
reject(Tensor(torch.tensor([0,9,1,9],dtype=torch.int32)[::2]),*base[1:])
reject(base[0],Tensor(torch.arange(12,dtype=torch.float32).reshape(2,6)[:,::2]),0,2,base[4])
reject(base[0],Tensor(torch._neg_view(w.value)),0,2,base[4])
reject(base[0],w,0,3,base[4])
reject(base[0],w,0,2,w)
reject(base[0],Tensor(w.value,data_ptr=lambda:w.data_ptr()+1),0,2,base[4])
reject(Tensor(base[0].value,numel=lambda:2**61),*base[1:])
# Original source failures: ID stride, column stride, output/input overlap.
ids_strided = torch.tensor([0,0,1,0],dtype=torch.int32)[::2]
out = torch.empty(2,3)
original(Tensor(ids_strided),w,0,2,Tensor(out))
assert not torch.equal(out,w.value)
column_weight = torch.arange(12,dtype=torch.float32).reshape(2,6)[:,::2]
original(base[0],Tensor(column_weight),0,2,Tensor(out))
assert not torch.equal(out,column_weight)
alias = torch.tensor([[10.],[20.]])
original(Tensor(torch.tensor([1,0],dtype=torch.int32)),Tensor(alias),0,2,Tensor(alias))
assert alias.tolist() == [[20.],[20.]]
# A masked tail default wraps 2^32 to zero and enables an invalid weight read.
try:
    original(Tensor(torch.tensor([-1],dtype=torch.int32)),Tensor(torch.empty(0,1)),0,2**32,Tensor(torch.empty(1,1)))
except IndexError:
    pass
else:
    raise AssertionError("original masked-default counterexample disappeared")
assert np.array(32768 * 65536, dtype=np.uint32).view(np.int32).item() == -(2**31)
# Source binding: keep the complete JIT body, exclusive interval, masks and casts visible.
python_tree = ast.parse((HERE / 'embedding_triton_kernel_repaired.py').read_text())
kernel_ast = next(n for n in python_tree.body if isinstance(n,ast.FunctionDef) and n.name=='embedding_kernel_repaired')
assert [ast.unparse(a.annotation) for a in kernel_ast.args.args[3:8]] == ['tl.int64']*5
casts = [n for n in ast.walk(kernel_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to']
assert len(casts)==4 and all(ast.unparse(n.args[0])=='tl.int64' for n in casts)
loop = kernel_ast.body[3]
assert ast.unparse(loop.iter)=='range(0, BLOCK_N, BLOCK_NN)'
assert [n.targets[0].id for n in loop.body if isinstance(n,ast.Assign)] == [
    'start_nn','offs_seq','n_ctx_mask','token_ids','id_mask','token_ids','dim_mask',
    'load_mask','store_mask','vecs','output_rows']
source = (HERE / 'Kernels.lean').read_text()
assert source.startswith('import VeriTile.Triton.DSL\nimport VeriTile.Triton.Source\n')
assert 'import bench.' not in source and 'Proofs' not in '\n'.join(source.splitlines()[:2])
assert 'input_ids = $index' in source and 'tl.float32' in source
assert 'other=$((0 : Int))' in source and 'token_ids < $((p.last : Int))' in source
assert 'id_mask = n_ctx_mask &' in source and 'output_rows = tl.where' in source
print(f'Embedding targeted CPU source/guard checks passed: {call_case.count} ordinary cases, '
      f'1 offset-view case, {reject.count} rejected inputs; no CUDA execution.')
