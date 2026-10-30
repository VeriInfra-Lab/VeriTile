"""Run the actual wrapper and stripped JIT body on bounded NumPy storage.

CUDA submission is mocked. Arithmetic is source-level FP32, not native-FMA
conformance. The original benchmark is parsed without running its GPU demo.
"""
import ast
import contextlib
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = ROOT / "tritonbench_g/apply_penalty/apply_penalty.py"
SOURCE = HERE / "apply_penalty_repaired.py"


class Tensor:
    def __init__(self, data, dtype=None, *, negative=False, device="cuda:0"):
        self.data = np.asarray(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.layout = "strided"
        self.device = SimpleNamespace(type=device.split(":")[0], name=device)
        self.negative = negative
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.pointer_override = None
        self.size_override = None

    def numel(self):
        return self.data.size if self.size_override is None else self.size_override

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def is_neg(self):
        return self.negative

    def data_ptr(self):
        return self.data.ctypes.data if self.pointer_override is None else self.pointer_override

    def tolist(self):
        return (-self.data if self.negative else self.data).tolist()

    def stride(self, dimension):
        return self.data.strides[dimension] // self.data.itemsize


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    __radd__ = __add__


class TL:
    constexpr = object()
    int64 = np.int64
    pid = 0
    writes = []

    @classmethod
    def program_id(cls, axis):
        assert axis == 0
        return cls.pid

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop, dtype=np.int64)

    @staticmethod
    def load(ptr, mask=True, other=0):
        offsets, masks = np.broadcast_arrays(ptr.offset, mask)
        assert np.all(offsets[masks] >= 0)
        assert np.all(offsets[masks] < ptr.tensor.data.size)
        result = np.full(offsets.shape, other, dtype=ptr.tensor.dtype)
        result[masks] = ptr.tensor.data.reshape(-1)[offsets[masks]]
        return result

    @classmethod
    def store(cls, ptr, value, mask=True):
        offsets, masks, values = np.broadcast_arrays(ptr.offset, mask, value)
        assert np.all(offsets[masks] >= 0)
        assert np.all(offsets[masks] < ptr.tensor.data.size)
        cls.writes.extend(offsets[masks].tolist())
        ptr.tensor.data.reshape(-1)[offsets[masks]] = values[masks]

    @staticmethod
    def where(condition, yes, no):
        return np.where(condition, yes, no)


class EraseIndexCasts(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            assert len(node.args) == 1 and ast.unparse(node.args[0]) == "tl.int64"
            return node.func.value
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "tl.cast":
            assert len(node.args) == 2 and ast.unparse(node.args[1]) == "tl.int64"
            return node.args[0]
        return node


class TritonCountPromotion(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name) and node.left.id == "batch_ids_count":
            node.left = ast.Call(func=ast.Name(id="fp32", ctx=ast.Load()), args=[node.left], keywords=[])
        return node


def function(path, name, erase_casts=False):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    if erase_casts:
        fn = EraseIndexCasts().visit(fn)
    return fn


def canonical_body(name):
    text = (HERE / "Kernels.lean").read_text()
    start = text.index(f"def {name}\n")
    start = text.index("ComputeKernel := triton {", start) + len("ComputeKernel := triton {")
    stop = text.index("\n}", start)
    body = "\n".join(line for line in text[start:stop].splitlines()
                     if not line.strip().startswith("tl.region "))
    body = re.sub(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", r"\1", body)
    return ast.parse(textwrap.dedent(body)).body


def same_body(left, right):
    module = lambda body: ast.Module(body=body, type_ignores=[])
    return ast.dump(module(left), include_attributes=False) == ast.dump(module(right), include_attributes=False)


def load_wrapper():
    original = function(ORIGINAL, "_fwd_kernel_apply_penalty")
    repaired = function(SOURCE, "_fwd_kernel_apply_penalty_repaired", True)
    original.name = repaired.name
    original.body = [n for n in original.body if not isinstance(n, ast.Return)]
    assert ast.dump(original, include_attributes=False) == ast.dump(repaired, include_attributes=False)
    assert same_body(canonical_body("kernel"), repaired.body), "Python/canonical kernel mismatch"
    reciprocal = function(SOURCE, "_fwd_kernel_apply_penalty_reciprocal", True)
    assert same_body(canonical_body("reciprocalKernel"), reciprocal.body), "Python/canonical candidate mismatch"
    candidate_repetition = next(n for n in reciprocal.body
                               if isinstance(n, ast.Assign) and n.targets[0].id == "rep_logits")
    assert ast.unparse(candidate_repetition.value) == (
        "tl.where(cur_logits > 0, cur_logits * (1 / cur_repetition), cur_logits * cur_repetition)")
    # Verify the candidate changes only the intended repetition expression.
    reciprocal.name = repaired.name
    candidate_repetition.value = next(n.value for n in repaired.body
                                     if isinstance(n, ast.Assign) and n.targets[0].id == "rep_logits")
    assert ast.dump(repaired, include_attributes=False) == ast.dump(reciprocal, include_attributes=False)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))
                assert kwargs["num_warps"] == 8 and kwargs["enable_fp_fusion"] is True
                for pid in range(grid[0]):
                    TL.pid = pid
                    with np.errstate(all="ignore"):
                        body(*(Pointer(x) if isinstance(x, Tensor) else x for x in args),
                             BLOCK_P=kwargs["BLOCK_P"])
            return launch

    # Triton promotes int32 counts to FP32 in this mixed multiplication;
    # NumPy would otherwise promote int32 * float32 arrays to float64.
    repaired = TritonCountPromotion().visit(repaired)
    env = {"tl": TL, "fp32": np.float32}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[repaired], type_ignores=[])), str(SOURCE), "exec"), env)
    body = env[repaired.name]
    torch = SimpleNamespace(Tensor=Tensor, strided="strided", float32=np.dtype("float32"),
                            int32=np.dtype("int32"), cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()),
                            no_grad=lambda: lambda f: f)
    import operator
    env.update(torch=torch, operator=operator,
               triton=SimpleNamespace(next_power_of_2=lambda x: 1 << (x-1).bit_length()),
               _fwd_kernel_apply_penalty_repaired=Kernel())
    definitions = [n for n in ast.parse(SOURCE.read_text()).body
                   if isinstance(n, ast.FunctionDef) and not n.name.startswith("_fwd_kernel_")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(SOURCE), "exec"), env)
    return env["apply_penalty_repaired"], env["source_block"], calls


def case(batches=2, vocab=8, length=3):
    logits = Tensor(np.arange(batches*vocab, dtype=np.float32).reshape(batches,vocab)-4)
    p = Tensor(np.full(batches, 1, np.float32))
    f = Tensor(np.full(batches, 2, np.float32))
    r = Tensor(np.full(batches, 2, np.float32))
    ids = Tensor(np.tile(np.arange(length, dtype=np.int32),batches))
    counts = Tensor(np.tile(np.arange(1,length+1,dtype=np.int32),batches))
    cumsum = Tensor(np.arange(batches+1,dtype=np.int32)*length)
    return [logits,p,f,r,ids,counts,cumsum,length]


def reference(args):
    logits,p,f,r,ids,counts,cumsum,_ = args
    out = logits.data.copy()
    for b in range(logits.shape[0]):
        for k in range(int(cumsum.data[b]),int(cumsum.data[b+1])):
            t,c = int(ids.data[k]),np.float32(counts.data[k])
            x = logits.data[b,t]
            with np.errstate(all="ignore"):
                repeated = np.float32(x/r.data[b] if x>0 else x*r.data[b])
                out[b,t] = np.float32(np.float32(repeated-np.float32(c*f.data[b]))-p.data[b])
    return out


def main():
    wrapper, block, calls = load_wrapper()
    passed = []

    def accept(name,args):
        before = [x.data.copy() for x in args[:7]]
        expected = reference(args)
        pointers = [x.data_ptr() for x in args[:7]]
        TL.writes.clear()
        old_calls = len(calls)
        assert wrapper(*args) is None
        np.testing.assert_allclose(args[0].data,expected,rtol=0,atol=0,equal_nan=True)
        for x,old in zip(args[1:7],before[1:]):
            np.testing.assert_array_equal(x.data,old)
        assert [x.data_ptr() for x in args[:7]] == pointers
        selected = [b*args[0].shape[1]+int(args[4].data[k]) for b in range(args[0].shape[0])
                    for k in range(int(args[6].data[b]),int(args[6].data[b+1]))]
        assert sorted(TL.writes) == sorted(selected) and len(set(TL.writes)) == len(TL.writes)
        if args[0].shape[0]:
            grid,native,options = calls[-1]
            assert grid == (args[0].shape[0],) and all(x is y for x,y in zip(native[:7],args[:7]))
            assert native[7:] == (args[0].shape[1],1)
        else:
            assert len(calls) == old_calls
        passed.append(name)

    def reject(name,args):
        before = [x.data.copy() for x in args[:7] if isinstance(x,Tensor)]
        old_calls = len(calls)
        try:
            wrapper(*args)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError("unexpected admission: "+name)
        assert len(calls) == old_calls
        for x,old in zip([x for x in args[:7] if isinstance(x,Tensor)],before):
            np.testing.assert_array_equal(x.data,old)
        passed.append(name)

    accept("positive negative zero and distinct rows",case())
    for length in [0,1,511,512,513,1024,1025,65535,65536]:
        accept("tile boundary "+str(length),case(1,max(1,length),length))
    accept("zero vocabulary",case(3,0,0))
    args=case(0,8,0); args[6]=Tensor([],np.int32); accept("zero batches empty cumsum",args)
    args=case(1,8,2); args[7]=0; accept("understated hint covered by block",args)
    args=case(); args[0]=Tensor(np.arange(30,dtype=np.float32)[3:19].reshape(2,8)); accept("nonzero storage offset",args)
    args=case(); args[2]=args[1]; accept("read-only penalty alias",args)
    args=case(); args[3].data[:]=0; accept("zero divisor structural admission",args)
    args=case(); args[0].data[0,:3]=[np.nan,np.inf,-np.inf]; accept("special FP values",args)
    args=case(); args[5].data[:]=2**24+1; accept("rounded int32 conversion",args)
    args=case(); args[4]=Tensor(np.r_[[-9],args[4].data,[-10]],np.int32); args[5]=Tensor(np.r_[[-5],args[5].data,[-7],[-8]],np.int32); args[6].data+=1; accept("ignored prefix suffix unequal capacities",args)
    args=case(1,8,1); args[0].data=args[0].data[:1,:]; accept("singleton dimensions",args)
    for label, mutate in [
        ("missing cumsum",lambda a:a.__setitem__(6,Tensor([0,3],np.int32))),
        ("short penalty",lambda a:a.__setitem__(1,Tensor([1],np.float32))),
        ("descending cumsum",lambda a:a[6].data.__setitem__(1,-1)),
        ("window beyond capacity",lambda a:a[6].data.__setitem__(2,100)),
        ("negative token",lambda a:a[4].data.__setitem__(0,-1)),
        ("token at vocabulary bound",lambda a:a[4].data.__setitem__(0,8)),
        ("negative count",lambda a:a[5].data.__setitem__(0,-1)),
        ("duplicate equal count",lambda a:(a[4].data.__setitem__(1,0),a[5].data.__setitem__(1,1))),
        ("duplicate conflicting count",lambda a:a[4].data.__setitem__(1,0)),
        ("negative hint",lambda a:a.__setitem__(7,-1)),
        ("hint resource bound",lambda a:a.__setitem__(7,65537)),
        ("noninteger hint",lambda a:a.__setitem__(7,2.5)),
        ("FP16 logits",lambda a:a.__setitem__(0,Tensor(a[0].data,np.float16))),
        ("int64 IDs",lambda a:a.__setitem__(4,Tensor(a[4].data,np.int64))),
        ("negative view",lambda a:setattr(a[0],"negative",True)),
        ("metadata negative view",lambda a:setattr(a[1],"negative",True)),
        ("CPU metadata",lambda a:setattr(a[1],"device",SimpleNamespace(type="cpu",name="cpu"))),
        ("device mismatch",lambda a:setattr(a[1],"device",SimpleNamespace(type="cuda",name="cuda:1"))),
        ("nontensor",lambda a:a.__setitem__(3,None)),
        ("wrong rank",lambda a:a.__setitem__(1,Tensor([[1,2]],np.float32))),
        ("extent int32 boundary",lambda a:setattr(a[0],"size_override",2**31)),
        ("noncontiguous metadata",lambda a:a.__setitem__(5,Tensor(np.arange(12,dtype=np.int32)[::2]))),
        ("noncontiguous logits",lambda a:a.__setitem__(0,Tensor(np.arange(32,dtype=np.float32).reshape(2,16)[:,::2]))),
        ("metadata logits alias",lambda a:a.__setitem__(1,Tensor(a[0].data.reshape(-1)[:2]))),
    ]:
        args=case(); mutate(args); reject(label,args)
    for role in [0,4]:
        for remainder in [1,2,3]:
            args=case(); args[role].pointer_override=args[role].data_ptr()+remainder
            reject("unaligned role %s remainder %s"%(role,remainder),args)
    args=case(1,513,513); args[7]=512; reject("truncation counterexample",args)
    args=case(0,0,0); args[0].pointer_override=1; accept("empty unaligned pointer unused",args)
    assert [block(x) for x in [0,1,511,512,513,600,1024,1025,65536]] == [512,512,512,512,1024,1024,1024,2048,65536]
    assert "Proofs" not in (HERE/"Kernels.lean").read_text()
    print("apply_penalty:",len(passed),"actual-wrapper/source cases passed; no GPU execution")


if __name__ == "__main__":
    main()
