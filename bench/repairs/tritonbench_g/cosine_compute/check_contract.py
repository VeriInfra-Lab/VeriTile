"""Source/contract and CPU storage replay; allocation/CUDA launch are mocked."""

import ast
import contextlib
import importlib.util
import math
from pathlib import Path
import re
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cosine_compute_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/cosine_compute/cosine_compute.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load(HERE.parent / "add_example/check_contract.py", "cosine_test_storage_helpers")
TensorFacade, Pointer, raw_storage = shared.TensorFacade, shared.Pointer, shared.raw_storage
function = shared.function


def source_checks():
    assert ast.dump(function(SOURCE, "cos_func")) == ast.dump(function(ORIGINAL, "cos_func"))
    expected = [
        "not isinstance(A, torch.Tensor)", "A.layout != torch.strided",
        "A.dtype != torch.float32", "not A.is_cuda", "not A.is_contiguous()",
        "A.is_neg()", "n_elements >= 2**31",
        "n_elements > 0 and A.data_ptr() % 4 != 0", "n_elements == 0",
    ]
    wrapper = function(SOURCE, "cos_repaired")
    guards = [node.test for node in wrapper.body if isinstance(node, ast.If)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(s, mode="eval").body) for s in expected]
    lean = (HERE / "Contract.lean").read_text()
    checks = lean.split("def checkTensorInput", 1)[1].split("def checkInput", 1)[0]
    assert re.findall(r"\bif (.*?) then", checks) == [
        "x.layout != .strided", "x.dtype != .floating .fp32", "!x.isCuda",
        "!x.isContiguous", "x.isNeg", "x.numel >= 2^31",
        "x.numel > 0 && x.dataPtr.toNat % 4 != 0",
    ]
    assert "| .nonTensor =>.error .tensor" in lean.replace("=> ", "=>")
    kernel_source = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in kernel_source
    assert "Region .fp32" in kernel_source and "def originalKernel" in kernel_source
    assert "def blockSize : Nat := 256" in kernel_source
    assert "def gridCount (n : Nat) : Nat := (n + 255) / 256" in kernel_source
    assert "FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert "Correct" not in (HERE / "FPEquiv.lean").read_text()
    # Pin the original finite lowering's five operations to the unchanged AST.
    native = (HERE.parents[2] / "tritonbench_g/cosine_compute/Proofs/NativeSource.lean").read_text()
    for register in ['"offset"', '"mask"', '"a_value"', '"b_value"']:
        assert register in native
    assert '.cosine (.toFP32 (.register "a_value"))' in native


def replay_kernel(kernel_name="cos_func"):
    pid = [0]

    def load_word(pointer, mask):
        value = torch.full_like(pointer.offsets, float("nan"), dtype=torch.float32)
        value[mask] = raw_storage(pointer.tensor)[pointer.tensor.storage_offset() + pointer.offsets[mask]]
        return value

    def store_word(pointer, value, mask):
        raw_storage(pointer.tensor)[pointer.tensor.storage_offset() + pointer.offsets[mask]] = value[mask]

    node = function(SOURCE, kernel_name)
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    namespace = {"tl": SimpleNamespace(program_id=lambda _: pid[0], arange=torch.arange,
                                       float32=torch.float32, cos=torch.cos,
                                       load=load_word, store=store_word)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace[kernel_name], pid


def host_checks(kernel_name="cos_func"):
    module = load(SOURCE, "cosine_contract_review")
    kernel, pid = replay_kernel(kernel_name)
    calls = []

    def empty_like(x):
        calls.append("allocation")
        return TensorFacade(torch.empty_like(x.data), str(x.device))

    class Launch:
        def __getitem__(self, grid):
            def run(x, out, n, block):
                calls.append((grid, x, out, n, block))
                assert grid == ((n + 255)//256, 1, 1) and block == 256
                # Reversed order exercises the proved complete-order freedom.
                for program in reversed(range(grid[0])):
                    pid[0] = program
                    kernel(Pointer(x.data), Pointer(out.data), n, block)
            return run

    module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
        empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.cos_func = Launch()
    bad = [([], TypeError), (TensorFacade(torch.eye(2).to_sparse()), ValueError),
           (TensorFacade(torch.ones(2, dtype=torch.float64)), TypeError),
           (TensorFacade(torch.ones(2), device="cpu", cuda=False), ValueError),
           (TensorFacade(torch.arange(6, dtype=torch.float32)[::2]), ValueError),
           (TensorFacade(torch._neg_view(torch.ones(2))), ValueError),
           (TensorFacade(torch.empty(2**31, device="meta")), ValueError)]
    bad += [(TensorFacade(shared.unaligned_tensor(i)), ValueError) for i in (1, 2, 3)]
    for x, error in bad:
        calls.clear()
        try:
            module.cos_repaired(x)
        except error:
            pass
        else:
            raise AssertionError("unsupported input admitted")
        assert calls == [], "rejection must precede allocation"

    backing = torch.arange(20, dtype=torch.float32)
    good = [torch.empty(0), torch.empty(2, 0), torch.tensor(-1.), backing[1:8],
            torch.as_strided(backing, (1, 4), (99, 1), 2),
            torch.as_strided(backing, (0,), (1,), 1000), shared.unaligned_tensor(1, empty=True),
            torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])]
    good += [torch.linspace(-6.0, 6.0, n) for n in (1, 255, 256, 257, 511, 512, 1024, 8192)]
    for tensor in good:
        calls.clear()
        x = TensorFacade(tensor)
        before = tensor.clone()
        result = module.cos_repaired(x)
        torch.testing.assert_close(result.data, torch.cos(tensor), rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(tensor, before, rtol=0, atol=0, equal_nan=True)
        assert result is not x and result.shape == x.shape and result.dtype == x.dtype
        assert result.device == x.device and result.is_contiguous()
        assert len(calls) == (2 if x.numel() else 1)
        if x.numel():
            assert calls[1][1] is x, "launch must use original storage"
    return len(bad), len(good)


def boundary_counterexamples():
    # Original empty host code: installed next_power_of_2(0) is zero.
    import triton
    block = triton.next_power_of_2(math.ceil(math.sqrt(0)))
    assert block == 0
    try:
        triton.cdiv(0, block)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("original empty failure not reproduced")
    # Signed i32 n_elements=-2**31 is below every signed i32 lane index.
    n = 2**31
    assert triton.next_power_of_2(math.ceil(math.sqrt(n))) == 65536
    assert all(not (offset < -2**31) for offset in (-2**31, -1, 0, 2**31-1))
    for n in (0, 1, 255, 256, 257, 2**31-1):
        grid = (n+255)//256
        assert grid*256 >= n
        if grid:
            assert (grid-1)*256+255 < 2**31
    # Original stride-two read: pointer lane one reads 1 rather than logical 2.
    backing = torch.arange(4, dtype=torch.float32)
    view = backing[::2]
    raw = raw_storage(view)[view.storage_offset():view.storage_offset()+view.numel()]
    assert not torch.equal(torch.cos(raw), torch.cos(view))



def check_candidate_binding():
    original = function(SOURCE, "cos_func")
    candidate = function(SOURCE, "cos_func_index_commuted")
    expected = ast.parse(ast.unparse(original)).body[0]
    expected.name = "cos_func_index_commuted"
    assignment = next(node for node in expected.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "offset")
    product = assignment.value.left
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
    source_checks()
    check_candidate_binding()
    rejected, admitted = host_checks()
    assert host_checks("cos_func_index_commuted") == (rejected, admitted)
    boundary_counterexamples()
    print(f"Passed source binding, {rejected} rejections, {admitted} admissions, and boundary counterexamples.")
    print("CPU backing-storage interpreter with mocked CUDA launch; no native GPU numerical claim.")
