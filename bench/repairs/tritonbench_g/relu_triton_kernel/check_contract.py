"""Focused source/guard binding and CPU storage checks. CUDA is mocked."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "relu_triton_kernel_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/relu_triton_kernel/relu_triton_kernel.py"
CHECKS = [
    ("x.layout != torch.strided", "x.layout != .strided", "layout : x.layout = .strided"),
    ("x.dtype != torch.float32", "x.dtype != .floating .fp32", "dtype : x.dtype = .floating .fp32"),
    ("not x.is_cuda", "!x.isCuda", 'cuda : x.device.kind = "cuda"'),
    ("not x.is_contiguous()", "!x.isContiguous", "contiguous : x.isContiguous = Bool.true"),
    ("x.is_neg()", "x.isNeg", "noNeg : x.isNeg = Bool.false"),
    ("N >= (1 << 31)", "x.numel >= 2^31", "extent : x.numel < 2^31"),
    ("N > 0 and x.data_ptr() % 4 != 0", "x.numel > 0 && x.dataPtr.toNat % 4 != 0",
     "alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0"),
]


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_binding():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == "2dffef82d69f280104c25b798e23f59a1ba9d00325552b91149a1d3b50fcffb0"
    original = function(ORIGINAL, "relu_kernel")
    assert isinstance(original.body[-1], ast.If)
    assert ast.unparse(original.body[-1].test) == "pid == 0"
    original.body[-1:] = original.body[-1].body
    assert ast.dump(original) == ast.dump(function(SOURCE, "relu_kernel"))
    wrapper = function(SOURCE, "relu")
    actual = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    expected = [ast.parse(x, mode="eval").body for x in
                ["not isinstance(x, torch.Tensor)", *(row[0] for row in CHECKS)]]
    assert list(map(ast.dump, actual)) == list(map(ast.dump, expected))
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(wrapper)) == len(CHECKS) + 1
    lean = (HERE / "Contract.lean").read_text()
    contract = lean.split("structure InputContract", 1)[1].split("inductive InputError", 1)[0]
    guards = lean.split("def checkTensorInputs", 1)[1].split("def checkInputs", 1)[0]
    assert re.findall(r"\bif (.*?) then", guards) == [row[1] for row in CHECKS]
    for _, _, premise in CHECKS:
        assert premise in contract
    assert "| .tensor x => checkTensorInputs x" in lean
    assert "| .nonTensor => .error .tensor" in lean
    source_text = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in source_text
    assert ".mk [" not in source_text
    # Pointer-coordinate specialization is the sole source-adapter step.
    class ElementCoordinates(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Constant) and node.left.value == 1:
                return node.right
            if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right
            return node
    for lean_name, python_name in (("kernel", "relu_kernel"), ("fusedKernel", "relu_inlined_kernel")):
        body = re.search(r"def " + lean_name + r"\b.*?triton \{\n(.*?)\n  \}", source_text, re.S).group(1)
        for name, value in (("ib", "0"), ("ob", "0"), ("elementBytes", "1"), ("N", "N"), ("B", "block_size")):
            body = body.replace("$(" + name + ")", value)
        body = "\n".join(line.strip() for line in body.splitlines())
        displayed = ElementCoordinates().visit(ast.parse(body)).body
        native = function(SOURCE, python_name).body
        if isinstance(native[0], ast.Expr) and isinstance(native[0].value, ast.Constant):
            native = native[1:]
        assert list(map(ast.dump, displayed)) == list(map(ast.dump, native))

    print("Source binding: original preserved; only kernel change is removing the pid-zero gate; all 8 guards match Lean.")


class TensorFacade:
    """Real CPU metadata and storage with the CUDA device query mocked."""
    def __init__(self, data, cuda=True):
        self.data, self.is_cuda = data, cuda
        self.device = torch.device("cuda:0" if cuda else "cpu")
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    @property
    def shape(self): return self.data.shape
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self):
        assert self.numel() > 0, "empty admission must not query a pointer"
        return 0 if self.data.device.type == "meta" else self.data.data_ptr()


def raw_storage(t):
    return torch.empty(0, dtype=torch.float32).set_(
        t.untyped_storage(), 0, (t.untyped_storage().nbytes() // 4,), (1,))


def unaligned(offset, empty=False):
    data = torch.arange(20, dtype=torch.uint8)
    return torch.empty(0, dtype=torch.float32).set_(
        data.untyped_storage()[offset:offset + 16], 0, (0 if empty else 4,), (1,))


class Pointer:
    def __init__(self, tensor, offsets=None): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, offsets)


def interpreter(path, name="relu_kernel"):
    pid = [0]
    def load(p, mask):
        out = torch.empty_like(p.offsets, dtype=torch.float32)
        out[mask] = raw_storage(p.tensor)[p.tensor.storage_offset() + p.offsets[mask]]
        return out
    def store(p, values, mask):
        raw_storage(p.tensor)[p.tensor.storage_offset() + p.offsets[mask]] = values[mask]
    node = function(path, name)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    namespace = {"tl": SimpleNamespace(program_id=lambda _: pid[0], arange=torch.arange,
                                        load=load, store=store, where=torch.where)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name], pid


def host_checks():
    spec = importlib.util.spec_from_file_location("relu_contract_check", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    kernel, pid = interpreter(SOURCE)
    def empty_like(x, dtype, device):
        assert dtype == torch.float32 and device == x.device
        calls.append("allocation")
        return TensorFacade(torch.empty_like(x.data))
    class Launch:
        def __getitem__(self, grid):
            def run(x, out, n, block):
                calls.append((grid, x, out, n, block))
                if x.data.device.type == "meta": return
                for program in range(grid[0]):
                    pid[0] = program
                    kernel(Pointer(x.data), Pointer(out.data), n, block)
            return run
    module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
        empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.relu_kernel = Launch()
    bad = [(None, TypeError, "tensor"),
           (TensorFacade(torch.eye(2).to_sparse()), TypeError, "strided"),
           (TensorFacade(torch.ones(2, dtype=torch.float64)), TypeError, "float32"),
           (TensorFacade(torch.ones(2), cuda=False), ValueError, "CUDA"),
           (TensorFacade(torch.ones(2, 3).T), ValueError, "contiguous"),
           (TensorFacade(torch.ones(6)[::2]), ValueError, "contiguous"),
           (TensorFacade(torch._neg_view(torch.ones(2))), ValueError, "negative view"),
           (TensorFacade(torch.empty(1 << 31, device="meta")), ValueError, "2**31")]
    for offset in (1, 2, 3):
        x = unaligned(offset)
        assert x.is_contiguous() and x.data_ptr() % 4 == offset
        bad.append((TensorFacade(x), ValueError, "aligned"))
    for x, error, message in bad:
        calls.clear()
        try: module.relu(x)
        except error as caught: assert message in str(caught)
        else: raise AssertionError("unsupported input admitted")
        assert not calls

    backing = torch.arange(-20., 20.)
    words = torch.tensor([0, 0x80000000, 0x7fc12345, 0xffc12345, 0x7f800001,
                          0x7f800000, 0xff800000, 1, 0x80000001, 0x3f800000, 0xbf800000],
                         dtype=torch.int64).to(torch.int32)
    special = words.view(torch.float32)
    good = [torch.randn(n) for n in (1, 1023, 1024, 1025, 2048, 2051)] + [
        torch.tensor(-2.), torch.empty(2, 0), backing[3:18], backing.reshape(4, 10),
        torch.as_strided(backing, (1, 4), (99, 1), 2), special,
        torch.as_strided(backing, (0,), (1,), 1000), unaligned(1, True)]
    for data in good:
        calls.clear()
        x = TensorFacade(data)
        before = raw_storage(data).view(torch.int32).clone()
        out = module.relu(x)
        expected = torch.where(data >= 0, data, 0.0)
        assert torch.equal(out.data.reshape(-1).view(torch.int32), expected.reshape(-1).view(torch.int32))
        assert torch.equal(raw_storage(data).view(torch.int32), before)
        assert out is not x and out.shape == x.shape and out.data.dtype == torch.float32
        assert out.is_contiguous() and not out.is_neg()
        if x.numel():
            assert len(calls) == 2
            grid, actual_x, actual_out, n, block = calls[1]
            assert actual_x is x and actual_out is out
            assert (grid, n, block) == (((x.numel()+1023)//1024,), x.numel(), 1024)
        else: assert calls == ["allocation"]

    # Trace the largest admitted size without allocating or executing it.
    calls.clear()
    large = TensorFacade(torch.empty((1 << 31)-1, device="meta"))
    returned = module.relu(large)
    grid, actual_x, actual_out, n, block = calls[1]
    assert actual_x is large and actual_out is returned
    assert (grid[0]-1)*block + 1023 == (1 << 31)-1
    assert n == (1 << 31)-1 and grid == (1 << 21,)
    print(f"Host/storage: {len(bad)} rejections, {len(good)} admitted cases, empty no-launch and metadata-only int32 boundary passed.")

    # Deterministic reproduction of the original output-coverage defect.
    old, old_pid = interpreter(ORIGINAL)
    x, out = torch.ones(1025), torch.full((1025,), -12345.)
    for program in (0, 1):
        old_pid[0] = program
        old(Pointer(x), Pointer(out), 1025, 1024)
    assert out[1023].item() == 1 and out[1024].item() == -12345
    print("Original defect reproduced: index 1024 retains allocation sentinel for 1025 ones.")
    print("CPU execution of kernel AST only; CUDA device/allocation/launch mocked. No native GPU claim.")



def candidate_checks():
    # Compare the real source expressions on full CPU storage, including tails.
    original = function(SOURCE, "relu_kernel")
    candidate = function(SOURCE, "relu_inlined_kernel")
    candidate.body = candidate.body[1:]  # documentation string
    assert list(map(ast.dump, original.body[:5])) == list(map(ast.dump, candidate.body[:5]))
    expected = ast.parse("tl.store(out_ptr + offsets, tl.where(x >= 0, x, 0.0), mask=mask)").body[0]
    assert ast.dump(candidate.body[-1]) == ast.dump(expected)
    lhs, lpid = interpreter(SOURCE)
    rhs, rpid = interpreter(SOURCE, "relu_inlined_kernel")
    words = torch.tensor([0, -2147483648, 1, -2147483647, 0x7f800000,
                          -8388608, 0x7fc12345, 0x7f800001, 0x3f800000,
                          -1082130432], dtype=torch.int32)
    for n in (0, 1, 1023, 1024, 1025, 2049):
        x = words.repeat((n + 9) // 10)[:n].view(torch.float32)
        a = torch.full((n + 7,), -12345.)
        b = a.clone()
        before = x.view(torch.int32).clone()
        for pid in range((n + 1023) // 1024):
            lpid[0] = rpid[0] = pid
            lhs(Pointer(x), Pointer(a), n, 1024)
            rhs(Pointer(x), Pointer(b), n, 1024)
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        assert torch.equal(x.view(torch.int32), before)
        assert torch.equal(a[n:], torch.full((7,), -12345.))
    print("Store-inline candidate: all-word samples, 6 boundaries, full output/frame equality.")

if __name__ == "__main__":
    source_binding()
    candidate_checks()
    host_checks()
