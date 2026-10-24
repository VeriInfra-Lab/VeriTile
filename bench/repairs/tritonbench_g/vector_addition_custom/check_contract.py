"""Targeted Python/Lean source binding and host-contract checks; no GPU claim."""

import ast
import contextlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "vector_addition_custom_repaired.py"
shared_spec = importlib.util.spec_from_file_location("custom_pair_helpers", HERE.parent / "add_example/check_contract.py")
shared = importlib.util.module_from_spec(shared_spec)
shared_spec.loader.exec_module(shared)
shared.SOURCE = SOURCE
ORIGINAL = HERE.parents[2] / "tritonbench_g/vector_addition_custom/vector_addition_custom.py"

# Classification precedes all attribute queries. The remaining rows bind each
# actual Python rejection to its Lean guard and named contract field.
TYPE_CHECK = "not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)"
CHECKS = [
    ("x.layout != torch.strided or y.layout != torch.strided",
     "x.layout != .strided || y.layout != .strided",
     "layout : x.layout = .strided ∧ y.layout = .strided"),
    ("x.dtype != torch.float32 or y.dtype != torch.float32",
     "x.dtype != .floating .fp32 || y.dtype != .floating .fp32",
     "dtype : x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32"),
    ("not x.is_cuda or not y.is_cuda",
     "!(x.isCuda && y.isCuda)", 'cuda : x.device.kind = "cuda" ∧ y.device.kind = "cuda"'),
    ("x.device != y.device", "x.device != y.device", "sameDevice : x.device = y.device"),
    ("x.shape != y.shape", "x.shape != y.shape", "sameShape : x.shape = y.shape"),
    ("x.ndim != 1", "x.shape.length != 1", "vector : x.shape.length = 1"),
    ("not x.is_contiguous() or not y.is_contiguous()",
     "!(x.isContiguous && y.isContiguous)",
     "contiguous : x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true"),
    ("x.is_neg() or y.is_neg()", "x.isNeg || y.isNeg",
     "noNeg : x.isNeg = Bool.false ∧ y.isNeg = Bool.false"),
    ("size >= (1 << 31)", "x.numel >= 2^31", "extent : x.numel < 2^31"),
    ("size > 0 and (x.data_ptr() % 4 != 0 or y.data_ptr() % 4 != 0)",
     "x.numel > 0 && (x.dataPtr.toNat % 4 != 0 || y.dataPtr.toNat % 4 != 0)",
     "alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ y.dataPtr.toNat % 4 = 0"),
]


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def check_source_binding():
    assert ast.dump(function(SOURCE, "_add_kernel")) == ast.dump(function(ORIGINAL, "_add_kernel"))
    shared.check_commuted_binding(SOURCE, "_add_kernel", "_add_commuted_kernel", ("a", "b"))
    wrapper = function(SOURCE, "custom_add_repaired")
    guards = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    expected = [ast.parse(condition, mode="eval").body
                for condition in [TYPE_CHECK, *(row[0] for row in CHECKS)]]
    assert list(map(ast.dump, guards)) == list(map(ast.dump, expected))
    lean = (HERE / "Contract.lean").read_text()
    contract = lean.split("structure InputContract", 1)[1].split("inductive InputError", 1)[0]
    checks = lean.split("def checkTensorInputs", 1)[1].split("def checkInputs", 1)[0]
    assert re.findall(r"\bif (.*?) then", checks) == [row[1] for row in CHECKS]
    for _, lean_guard, premise in CHECKS:
        _, checks = checks.split(lean_guard, 1)
        assert premise in contract
    classification = lean.split("def checkInputs", 1)[1].split("def AdmittedArguments", 1)[0]
    assert "match x, y with" in classification
    assert "| .tensor tx, .tensor ty => checkTensorInputs tx ty" in classification
    assert "| _, _ => .error .tensors" in classification
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(wrapper)) == 1 + len(CHECKS)

    # The source module may reuse another source, but must not import proofs.
    pending, visited = [HERE / "Kernels.lean"], set()
    root = HERE.parents[3]
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        for line in source.read_text().splitlines():
            if line.startswith("import "):
                for module in line.split()[1:]:
                    assert ".Proofs." not in module, (source, module)
                    if module.startswith("bench."):
                        pending.append(root / (module.replace(".", "/") + ".lean"))



class TensorFacade:
    """Real CPU tensor metadata with a mocked CUDA owner for host tests."""

    def __init__(self, data, device="cuda:0", cuda=True):
        self.data, self.device, self.is_cuda = data, torch.device(device), cuda

    @property
    def layout(self): return self.data.layout

    @property
    def dtype(self): return self.data.dtype

    @property
    def shape(self): return self.data.shape

    @property
    def ndim(self): return self.data.ndim

    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()

    def data_ptr(self):
        assert self.numel() > 0, "empty admission must not query the data pointer"
        return self.data.data_ptr()


def unaligned_tensor(byte_offset, empty=False):
    """A real contiguous CPU FP32 view backed by a byte-offset storage slice."""
    backing = torch.arange(20, dtype=torch.uint8)
    storage = backing.untyped_storage()[byte_offset:byte_offset + 16]
    return torch.empty(0, dtype=torch.float32).set_(
        storage, 0, (0 if empty else 4,), (1,))


def raw_storage(tensor):
    """Physical FP32 storage, without Tensor view/negative interpretation."""
    return torch.empty(0, dtype=torch.float32).set_(
        tensor.untyped_storage(), 0,
        (tensor.untyped_storage().nbytes() // 4,), (1,))


class Pointer:
    def __init__(self, tensor, offsets=None):
        self.tensor = tensor
        self.offsets = offsets

    def __add__(self, offsets):
        return Pointer(self.tensor, offsets)


def kernel_cpu_interpreter():
    """Execute the unchanged kernel AST over real CPU backing storage."""
    current_pid = [0]

    def load(pointer, mask):
        result = torch.zeros_like(pointer.offsets, dtype=torch.float32)
        storage = raw_storage(pointer.tensor)
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask]
        result[mask] = storage[indices]
        return result

    def store(pointer, value, mask):
        storage = raw_storage(pointer.tensor)
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask]
        storage[indices] = value[mask]

    kernel = function(SOURCE, "_add_kernel")
    kernel.decorator_list = []
    for argument in kernel.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[]))
    namespace = {"tl": SimpleNamespace(program_id=lambda axis: current_pid[0],
                                       arange=torch.arange, load=load, store=store)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_add_kernel"], current_pid


def check_tensor_semantics():
    backing = torch.arange(12, dtype=torch.float32)
    tensors = [backing, backing[2:7], backing.reshape(3, 4), torch.tensor(-2.),
               torch.as_strided(backing, (1, 4), (99, 1), 2)]
    for tensor in tensors:
        assert tensor.is_contiguous() and not tensor.is_neg()
        assert tensor.data_ptr() == tensor.untyped_storage().data_ptr() + 4*tensor.storage_offset()
        physical = raw_storage(tensor)[tensor.storage_offset():tensor.storage_offset()+tensor.numel()]
        torch.testing.assert_close(physical, tensor.reshape(-1), rtol=0, atol=0)
    negative = torch._neg_view(backing[2:7])
    assert negative.is_contiguous() and negative.is_neg()
    assert not torch.equal(raw_storage(negative)[2:7], negative)
    empty = torch.as_strided(backing, (0,), (1,), 1000)
    assert empty.is_contiguous() and empty.data_ptr() == 0
    assert empty.storage_offset() > empty.untyped_storage().nbytes() // 4
    for offset in (1, 2, 3):
        unaligned = unaligned_tensor(offset)
        assert unaligned.dtype == torch.float32 and unaligned.layout == torch.strided
        assert unaligned.is_contiguous() and not unaligned.is_neg()
        assert unaligned.storage_offset() == 0
        assert unaligned.data_ptr() % 4 == offset
    empty_unaligned = unaligned_tensor(1, empty=True)
    assert empty_unaligned.is_contiguous() and empty_unaligned.numel() == 0
    assert empty_unaligned.untyped_storage().data_ptr() % 4 == 1
    print("Tensor API checks: offsets, singleton strides, negative views, empty tensors and byte-unaligned storage passed.")


def check_host():
    spec = importlib.util.spec_from_file_location("vector_addition_custom_contract_review", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    kernel, current_pid = kernel_cpu_interpreter()

    def empty_like(x):
        calls.append("allocation")
        return TensorFacade(torch.full_like(x.data, float("nan")), str(x.device))

    class Launch:
        def __getitem__(self, grid):
            def run(x, y, out, n, BLOCK):
                block = BLOCK
                calls.append((grid, x, y, out, n, block))
                for pid in range(grid[0]):
                    current_pid[0] = pid
                    kernel(Pointer(x.data), Pointer(y.data), Pointer(out.data), n, block)
            return run

    module.torch = SimpleNamespace(
        Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
        empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module._add_kernel = Launch()
    valid = TensorFacade(torch.ones(3))
    negative = TensorFacade(torch._neg_view(torch.ones(3)))
    assert negative.is_contiguous() and negative.is_neg()
    bad = [
        ([], valid, TypeError, "tensors"),
        (TensorFacade(torch.eye(3).to_sparse()), valid, TypeError, "strided"),
        (TensorFacade(torch.ones(3, dtype=torch.float64)), valid, TypeError, "float32"),
        (TensorFacade(torch.ones(3), device="cpu", cuda=False), valid, ValueError, "CUDA"),
        (TensorFacade(torch.ones(3), device="cuda:1"), valid, ValueError, "same device"),
        (TensorFacade(torch.ones(1, 3)), valid, ValueError, "same shape"),
        (TensorFacade(torch.ones(6)[::2]), valid, ValueError, "contiguous"),
        (valid, TensorFacade(torch.ones(6)[::2]), ValueError, "contiguous"),
        (negative, valid, ValueError, "negative view"),
        (valid, negative, ValueError, "negative view"),
    ]
    large = TensorFacade(torch.empty(1 << 31, device="meta"))
    bad.append((large, large, ValueError, "smaller than 2**31"))
    for shape in [(), (2, 2), (0, 2)]:
        rank = TensorFacade(torch.empty(shape))
        bad.append((rank, rank, ValueError, "one-dimensional"))
    aligned = TensorFacade(torch.ones(4))
    for offset in (1, 2, 3):
        unaligned = TensorFacade(unaligned_tensor(offset))
        bad.extend([
            (unaligned, aligned, ValueError, "4-byte-aligned"),
            (aligned, unaligned, ValueError, "4-byte-aligned"),
        ])
    for x, y, error, message in bad:
        calls.clear()
        try:
            module.custom_add_repaired(x, y)
        except error as caught:
            assert message in str(caught)
        else:
            raise AssertionError("out-of-contract input was accepted")
        assert not calls, "rejection must precede allocation and launch"

    backing = torch.arange(12, dtype=torch.float32)
    good = [
        *[(torch.arange(n, dtype=torch.float32), torch.ones(n)) for n in [0, 1, 15, 16, 17, 31, 32, 33, 1025]],
        (torch.as_strided(backing, (1,), (99,), 2), torch.ones(1)),
        (torch.ones(5), torch.ones(5)),
        (torch.as_strided(backing, (0,), (1,), 1000), torch.empty(0)),
        (unaligned_tensor(1, empty=True), unaligned_tensor(3, empty=True)),
        (backing[2:7], backing[4:9]),
        (backing[1:6], backing[3:8]),
        (backing, backing),
        (torch.tensor([float("nan"), float("inf"), -float("inf")]), torch.ones(3)),
    ]
    for a, b in good:
        calls.clear()
        x, y = TensorFacade(a), TensorFacade(b)
        before_x, before_y = a.clone(), b.clone()
        out = module.custom_add_repaired(x, y)
        torch.testing.assert_close(out.data, a + b, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(a, before_x, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(b, before_y, rtol=0, atol=0, equal_nan=True)
        assert out is not x and out is not y
        assert out.shape == x.shape and out.data.is_contiguous()
        if x.numel():
            assert len(calls) == 2
            grid, actual_x, actual_y, actual_out, n, block = calls[1]
            assert actual_x is x and actual_y is y and actual_out is out
            assert (grid, n, block) == (((x.numel() + 15) // 16,), x.numel(), 16)
        else:
            assert calls == ["allocation"]
    print(f"Eleven source-bound checks; {len(bad)} rejection and {len(good)} admission cases passed.")
    print("Original kernel AST executed over CPU storage; CUDA calls mocked, no GPU execution.")


def check_original_counterexamples():
    # Run the actual original wrapper AST and unchanged kernel over CPU storage.
    wrapper = function(ORIGINAL, "custom_add")
    wrapper.decorator_list = []
    kernel, current_pid = kernel_cpu_interpreter()
    allocated = []

    def empty_like(x):
        out = torch.full_like(x, float("nan"))
        allocated.append(out)
        return out

    class Launch:
        def __getitem__(self, grid):
            def run(x, y, out, size, BLOCK):
                for pid in range(grid[0]):
                    current_pid[0] = pid
                    kernel(Pointer(x), Pointer(y), Pointer(out), size, BLOCK)
            return run

    namespace = {"torch": SimpleNamespace(empty_like=empty_like),
                 "triton": SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b),
                 "_add_kernel": Launch()}
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, str(ORIGINAL), "exec"), namespace)
    original = namespace["custom_add"]
    x = torch.arange(6, dtype=torch.float32)[::2]
    y = torch.ones(3)
    actual = original(x, y)
    assert actual.tolist() == [1., 2., 3.]
    assert (x + y).tolist() == [1., 3., 5.]
    x = torch.ones(2, 2)
    actual = original(x, x)
    assert actual.reshape(-1)[:2].tolist() == [2., 2.]
    assert torch.isnan(actual.reshape(-1)[2:]).all()
    x = torch._neg_view(torch.tensor([1., 2., 3.]))
    actual = original(x, torch.ones(3))
    assert actual.tolist() == [2., 3., 4.]
    assert (x + 1).tolist() == [0., -1., -2.]
    # Signed extent at 2**31 is INT_MIN; every signed lane fails the mask.
    extent = -(1 << 31)
    for pid in [0, 1, (1 << 27) - 1]:
        for lane in range(16):
            offset = ((pid * 16 + lane + (1 << 31)) % (1 << 32)) - (1 << 31)
            assert not offset < extent
    print("Original defects reproduced: strided and negative vectors, multidimensional unwritten tail; int32 boundary checked.")


if __name__ == "__main__":
    check_source_binding()
    check_tensor_semantics()
    check_host()
    check_original_counterexamples()
    shared.check_kernel_pair(16, "_add_kernel", "_add_commuted_kernel")
