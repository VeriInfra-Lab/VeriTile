"""Targeted source, guard, storage and host checks; CUDA calls are mocked."""
import ast
import contextlib
import ctypes
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sin_kernel_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/sin_kernel/sin_kernel.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the existing CPU Tensor/pointer fixtures and storage reader.
fixtures = load_module("add_contract_fixtures", HERE.parent / "add_example/check_contract.py")
TensorFacade = fixtures.TensorFacade
Pointer = fixtures.Pointer
raw_storage = fixtures.raw_storage
unaligned_tensor = fixtures.unaligned_tensor
function = fixtures.function

CHECKS = [
    ("not isinstance(x, torch.Tensor)", None, None),
    ("x.layout != torch.strided", "x.layout != .strided", "layout : x.layout = .strided"),
    ("x.dtype != torch.float32", "x.dtype != .floating .fp32", "dtype : x.dtype = .floating .fp32"),
    ("not x.is_cuda", "!x.isCuda", 'cuda : x.device.kind = "cuda"'),
    ("not x.is_contiguous()", "!x.isContiguous", "contiguous : x.isContiguous = Bool.true"),
    ("x.is_neg()", "x.isNeg", "noNeg : x.isNeg = Bool.false"),
    ("n_elements >= (1 << 31)", "x.numel >= 2^31", "extent : x.numel < 2^31"),
    ("n_elements > 0 and x.data_ptr() % 4 != 0", "x.numel > 0 && x.dataPtr.toNat % 4 != 0",
     "alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0"),
]


def check_source_binding():
    assert ast.dump(function(SOURCE, "kernel_function")) == ast.dump(function(ORIGINAL, "kernel_function"))
    wrapper = function(SOURCE, "call_kernel")
    guards = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    assert [ast.dump(g) for g in guards] == [ast.dump(ast.parse(c, mode="eval").body) for c, _, _ in CHECKS]
    lean = (HERE / "Contract.lean").read_text()
    contract = lean.split("structure InputContract", 1)[1].split("inductive InputError", 1)[0]
    checks = lean.split("def checkTensorInputs", 1)[1].split("def checkInputs", 1)[0]
    assert re.findall(r"\bif (.*?) then", checks) == [c for _, c, _ in CHECKS[1:]]
    assert all(p in contract for _, _, p in CHECKS[1:])
    assert "| .tensor x => checkTensorInputs x" in lean
    assert "| _ => .error .tensor" in lean
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(wrapper)) == len(CHECKS)
    calls = [ast.unparse(n.func) for n in ast.walk(wrapper) if isinstance(n, ast.Call)]
    assert not any(c.endswith((".clone", ".contiguous", ".resolve_neg", ".flatten")) for c in calls)
    source = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in source
    assert "def originalKernel" in source and "Region .fp32" in source
    assert "Source.source_program_reduction" in (HERE / "Proofs/Execution.lean").read_text()
    assert "sourceKernel" in (HERE / "Proofs/Source.lean").read_text()
    assert "fp32Kernel n" in (HERE / "Proofs/FP.lean").read_text()


def interpreter(kernel_name="kernel_function"):
    pid = [0]

    def load(pointer, mask):
        # Inactive loads may have arbitrary values; use NaN to exercise masking.
        values = torch.full_like(pointer.offsets, float("nan"), dtype=torch.float32)
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask]
        values[mask] = raw_storage(pointer.tensor)[indices]
        return values

    def store(pointer, values, mask):
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask]
        raw_storage(pointer.tensor)[indices] = values[mask]

    kernel = function(SOURCE, kernel_name)
    kernel.decorator_list = []
    for arg in kernel.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[]))
    ns = {"tl": SimpleNamespace(program_id=lambda _: pid[0], arange=torch.arange,
                                 load=load, store=store, math=SimpleNamespace(sin=torch.sin))}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns[kernel_name], pid


def check_host(kernel_name="kernel_function"):
    module = load_module("sin_contract_review", SOURCE)
    kernel, pid = interpreter(kernel_name)
    calls = []
    allocations = []

    @contextlib.contextmanager
    def device(context):
        calls.append(("device", context))
        yield

    def empty_like(x):
        calls.append("allocation")
        output = TensorFacade(torch.empty_like(x.data), str(x.device))
        allocations.append(output)
        return output

    class Launch:
        def __getitem__(self, grid):
            def run(x, output, n_elements, BLOCK_SIZE):
                calls.append((grid, x, output, n_elements, BLOCK_SIZE))
                for job in reversed(range(grid[0])):
                    pid[0] = job
                    kernel(Pointer(x.data), Pointer(output.data), n_elements, BLOCK_SIZE)
            return run

    module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
                                   empty_like=empty_like, cuda=SimpleNamespace(device=device))
    module.kernel_function = Launch()
    bad = [([], TypeError, "tensor"),
           (TensorFacade(torch.eye(3).to_sparse()), TypeError, "strided"),
           (TensorFacade(torch.ones(3, dtype=torch.float64)), TypeError, "float32"),
           (TensorFacade(torch.ones(3, dtype=torch.bfloat16)), TypeError, "float32"),
           (TensorFacade(torch.ones(3), device="cpu", cuda=False), ValueError, "CUDA"),
           (TensorFacade(torch.arange(6.)[::2]), ValueError, "contiguous"),
           (TensorFacade(torch._neg_view(torch.arange(3.))), ValueError, "negative view"),
           (TensorFacade(torch.empty(1 << 31, device="meta")), ValueError, "smaller than 2**31")]
    bad += [(TensorFacade(unaligned_tensor(i)), ValueError, "4-byte-aligned") for i in (1, 2, 3)]
    for x, error, text in bad:
        calls.clear()
        try:
            module.call_kernel(x)
        except error as caught:
            assert text in str(caught)
        else:
            raise AssertionError("unsupported input accepted")
        assert not calls, "rejection must precede allocation/device entry"

    base = torch.arange(1100, dtype=torch.float32) / 32
    good = [torch.empty(0), torch.empty(2, 0), torch.tensor(-2.), torch.arange(5.),
            torch.arange(1024.)/100, base[:1025], base[3:1032], base.reshape(20, 55),
            torch.as_strided(base, (1, 4), (99, 1), 2),
            torch.as_strided(base, (0,), (1,), 10000), unaligned_tensor(1, empty=True),
            torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0.])]
    for data in good:
        calls.clear()
        x = TensorFacade(data, device="cuda:1")
        before = data.clone()
        output = module.call_kernel(x)
        torch.testing.assert_close(output.data, torch.sin(data), rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(data, before, rtol=0, atol=0, equal_nan=True)
        assert output is allocations[-1] and output is not x
        assert output.shape == x.shape and output.device == x.device
        assert output.dtype == torch.float32 and output.is_contiguous()
        assert calls[:2] == [("device", x.device), "allocation"]
        if x.numel():
            assert len(calls) == 3
            grid, actual, returned, n, block = calls[-1]
            assert actual is x and returned is output
            assert (grid, n, block) == (((x.numel()+1023)//1024,), x.numel(), 1024)
        else:
            assert len(calls) == 2
    print(f"Source AST and eight guards: {len(bad)} rejection and {len(good)} admission cases passed.")


def check_original_counterexamples():
    kernel, pid = interpreter()
    for data in (torch.arange(8.)[::2], torch._neg_view(torch.tensor([0.2, 0.4]))):
        out = torch.empty_like(data)
        pid[0] = 0
        kernel(Pointer(data), Pointer(out), data.numel(), 1024)
        assert not torch.allclose(out, torch.sin(data))
    # Original signed int32 extent is the minimum signed value at this count.
    i32 = lambda n: ctypes.c_int32(n).value
    assert i32(1 << 31) == -(1 << 31)
    for n in (0, 1, 1023, 1024, 1025, (1 << 31)-1):
        grid = (n+1023)//1024
        for job in set((0, max(0, grid-1))):
            if job < grid:
                for lane in (0, 1023):
                    offset = job*1024 + lane
                    assert 0 <= offset < (1 << 31)
                    assert (i32(offset) < i32(n)) == (offset < n)
    for job in (0, (1 << 21)-1):
        assert all(not (i32(job*1024+lane) < i32(1 << 31)) for lane in range(1024))
    print("Original stride/negative-view counterexamples and signed-index boundaries passed.")



def check_candidate_binding():
    original = function(SOURCE, "kernel_function")
    candidate = function(SOURCE, "kernel_function_index_commuted")
    expected = ast.parse(ast.unparse(original)).body[0]
    expected.name = "kernel_function_index_commuted"
    assignment = next(node for node in expected.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "block_start")
    product = assignment.value
    assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
    product.left, product.right = product.right, product.left
    assert ast.dump(candidate) == ast.dump(expected)
    canonical = (HERE / "Kernels.lean").read_text()
    assert "def kernel " in canonical and "def indexCommutedKernel " in canonical
    assert "Region .fp32" in canonical and "blockSize" in canonical
    assert ".Proofs." not in canonical
    public = (HERE / "FPEquiv.lean").read_text()
    assert "assumptions : Spec.Assumptions FP.GuardedFragment := []" in public
    assert "domain := []" in public and "Host" not in public
    assert "originalKernel" in (HERE / "Correct.lean").read_text() or "kernel X O n blockSize" in (HERE / "Correct.lean").read_text()

if __name__ == "__main__":
    check_source_binding()
    check_candidate_binding()
    fixtures.check_tensor_semantics()
    check_host()
    check_host("kernel_function_index_commuted")
    check_original_counterexamples()
    print(f"CPU storage execution only; CUDA mocked. CUDA available: {torch.cuda.is_available()}.")
