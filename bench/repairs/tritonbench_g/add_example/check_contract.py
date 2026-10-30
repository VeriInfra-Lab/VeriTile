"""Targeted Python/Lean source binding and host-contract checks; no GPU claim."""

import ast
import contextlib
import copy
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "add_example_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/add_example/add_example.py"

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
    ("not x.is_contiguous() or not y.is_contiguous()",
     "!(x.isContiguous && y.isContiguous)",
     "contiguous : x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true"),
    ("x.is_neg() or y.is_neg()", "x.isNeg || y.isNeg",
     "noNeg : x.isNeg = Bool.false ∧ y.isNeg = Bool.false"),
    ("n_elements >= (1 << 31)", "x.numel >= 2^31", "extent : x.numel < 2^31"),
    ("n_elements > 0 and (x.data_ptr() % 4 != 0 or y.data_ptr() % 4 != 0)",
     "x.numel > 0 && (x.dataPtr.toNat % 4 != 0 || y.dataPtr.toNat % 4 != 0)",
     "alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ y.dataPtr.toNat % 4 = 0"),
]


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def check_source_binding():
    assert ast.dump(function(SOURCE, "add_kernel")) == ast.dump(function(ORIGINAL, "add_kernel"))
    check_commuted_binding(SOURCE)
    wrapper = function(SOURCE, "add_wrapper")
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



def check_commuted_binding(source, original="add_kernel", candidate="add_commuted_kernel", operands=("x", "y")):
    expected = copy.deepcopy(function(source, original))
    expected.name = candidate
    additions = [node for node in ast.walk(expected) if isinstance(node, ast.BinOp)
                 and isinstance(node.op, ast.Add)
                 and isinstance(node.left, ast.Name) and node.left.id == operands[0]
                 and isinstance(node.right, ast.Name) and node.right.id == operands[1]]
    assert len(additions) == 1
    additions[0].left, additions[0].right = additions[0].right, additions[0].left
    assert ast.dump(function(source, candidate)) == ast.dump(expected)


def check_kernel_pair(block, original="add_kernel", candidate="add_commuted_kernel"):
    cases = 0
    for inactive in (float("nan"), float("inf"), -float("inf")):
        lhs, left_pid = kernel_cpu_interpreter(original, inactive)
        rhs, right_pid = kernel_cpu_interpreter(candidate, inactive)
        for n in (0, 1, block-1, block, block+1, 2*block+1):
            x = torch.arange(n, dtype=torch.float32) - 13
            y = torch.arange(n, dtype=torch.float32) * -0.5
            a, b = torch.empty_like(x), torch.empty_like(x)
            for pid in reversed(range(max(1, (n+block-1)//block))):
                left_pid[0] = right_pid[0] = pid
                lhs(Pointer(x), Pointer(y), Pointer(a), n, block)
                rhs(Pointer(x), Pointer(y), Pointer(b), n, block)
            torch.testing.assert_close(a, x+y, rtol=0, atol=0)
            torch.testing.assert_close(b, a, rtol=0, atol=0)
            cases += 1
    print(f"{cases} kernel-pair CPU cases passed: empty/full/tail masks and NaN/Inf inactive loads.")


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


def kernel_cpu_interpreter(kernel_name="add_kernel", inactive=0.0):
    """Execute the unchanged kernel AST over real CPU backing storage."""
    current_pid = [0]

    def load(pointer, mask):
        result = torch.full_like(pointer.offsets, inactive, dtype=torch.float32)
        storage = raw_storage(pointer.tensor)
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask]
        result[mask] = storage[indices]
        return result

    def store(pointer, value, mask):
        storage = raw_storage(pointer.tensor)
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask]
        storage[indices] = value[mask]

    kernel = function(SOURCE, kernel_name)
    kernel.decorator_list = []
    for argument in kernel.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[]))
    namespace = {"tl": SimpleNamespace(program_id=lambda axis: current_pid[0],
                                       arange=torch.arange, load=load, store=store)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[kernel_name], current_pid


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
    spec = importlib.util.spec_from_file_location("add_example_contract_review", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    kernel, current_pid = kernel_cpu_interpreter()

    def zeros_like(x):
        calls.append("allocation")
        return TensorFacade(torch.zeros_like(x.data), str(x.device))

    class Launch:
        def __getitem__(self, grid):
            def run(x, y, out, n, block):
                calls.append((grid, x, y, out, n, block))
                for pid in range(grid[0]):
                    current_pid[0] = pid
                    kernel(Pointer(x.data), Pointer(y.data), Pointer(out.data), n, block)
            return run

    module.torch = SimpleNamespace(
        Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
        zeros_like=zeros_like, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.add_kernel = Launch()
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
            module.add_wrapper(x, y)
        except error as caught:
            assert message in str(caught)
        else:
            raise AssertionError("out-of-contract input was accepted")
        assert not calls, "rejection must precede allocation and launch"

    backing = torch.arange(12, dtype=torch.float32)
    good = [
        (torch.ones(5), torch.ones(5)),
        (torch.ones(2, 3), torch.ones(2, 3)),
        (torch.tensor(2.), torch.tensor(-3.)),
        (torch.empty(2, 0), torch.empty(2, 0)),
        (torch.as_strided(backing, (0,), (1,), 1000), torch.empty(0)),
        (unaligned_tensor(1, empty=True), unaligned_tensor(3, empty=True)),
        (torch.as_strided(backing, (1, 4), (99, 1), 2), torch.ones(1, 4)),
        (backing[2:7], backing[4:9]),
        (backing[1:6], backing[3:8]),
        (backing, backing),
        (torch.tensor([float("nan"), float("inf"), -float("inf")]), torch.ones(3)),
    ]
    for a, b in good:
        calls.clear()
        x, y = TensorFacade(a), TensorFacade(b)
        before_x, before_y = a.clone(), b.clone()
        out = module.add_wrapper(x, y)
        torch.testing.assert_close(out.data, a + b, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(a, before_x, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(b, before_y, rtol=0, atol=0, equal_nan=True)
        assert out is not x and out is not y
        assert out.shape == x.shape and out.data.is_contiguous()
        if x.numel():
            assert len(calls) == 2
            grid, actual_x, actual_y, actual_out, n, block = calls[1]
            assert actual_x is x and actual_y is y and actual_out is out
            assert (grid, n, block) == (((x.numel() + 3) // 4,), x.numel(), 4)
        else:
            assert calls == ["allocation"]
    print(f"Ten source-bound checks; {len(bad)} rejection and {len(good)} admission cases passed.")
    print("Original kernel AST executed over CPU storage; CUDA calls mocked, no GPU execution.")


if __name__ == "__main__":
    check_source_binding()
    check_tensor_semantics()
    check_host()
    check_kernel_pair(4)
