"""Targeted source/contract checks using the existing addition CPU interpreter."""

import ast
import contextlib
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent.parent
SOURCE = HERE / "vector_addition_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/vector_addition/vector_addition.py"
SHARED = HERE.parent / "add_example"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_module("addition_contract_helpers", SHARED / "check_contract.py")
shared.SOURCE = SOURCE
Tensor = shared.TensorFacade


def source_binding():
    assert ast.dump(shared.function(SOURCE, "add_kernel")) == ast.dump(shared.function(ORIGINAL, "add_kernel"))
    shared.check_commuted_binding(SOURCE)
    wrapper = shared.function(SOURCE, "add")
    guards = [node.test for node in wrapper.body if isinstance(node, ast.If)]
    conditions = [shared.TYPE_CHECK, *(row[0] for row in shared.CHECKS)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(s, mode="eval").body) for s in conditions]
    contract = (SHARED / "Contract.lean").read_text()
    for _, guard, field in shared.CHECKS:
        assert guard in contract and field in contract
    local_contract = (HERE / "Contract.lean").read_text()
    assert "export VeriTile.Bench.Repairs.AddExample.Contiguous" in local_contract
    assert "InputContract InputError checkTensorInputs" in local_contract
    calls = [ast.unparse(n.func) for n in ast.walk(wrapper) if isinstance(n, ast.Call)]
    assert "torch.empty_like" in calls
    assert not any(word in " ".join(calls) for word in ("clone", "resolve_neg", "contiguous(", "flatten"))
    assert "BLOCK_SIZE = 1024" in ast.unparse(wrapper)
    kernels = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in kernels
    assert 'add_kernel "x" "y" "out" n blockSize' in kernels and 'def blockSize : Nat := 1024' in kernels and 'Region .fp32' in kernels
    for name, forbidden in (("Correct.lean", "FPEquiv"), ("FPEquiv.lean", "Correct")):
        assert not any(forbidden in line for line in (HERE / name).read_text().splitlines() if line.startswith("import "))


def host_and_counterexamples():
    module = load_module("vector_addition_revision", SOURCE)
    kernel, current_pid = shared.kernel_cpu_interpreter()
    calls = []

    def empty_like(x):
        calls.append("allocation")
        return Tensor(torch.empty_like(x.data), str(x.device))

    class Launch:
        def __getitem__(self, grid):
            def run(x, y, out, n, block):
                calls.append((grid, x, y, out, n, block))
                for pid in range(grid[0]):
                    current_pid[0] = pid
                    kernel(shared.Pointer(x.data), shared.Pointer(y.data), shared.Pointer(out.data), n, block)
            return run

    module.torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
        empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.add_kernel = Launch()
    valid = Tensor(torch.ones(3))
    negative = Tensor(torch._neg_view(torch.ones(3)))
    bad = [([], valid), (Tensor(torch.eye(3).to_sparse()), valid),
        (Tensor(torch.ones(3, dtype=torch.float16)), valid),
        (Tensor(torch.ones(3), device="cpu", cuda=False), valid),
        (Tensor(torch.ones(3), device="cuda:1"), valid),
        (Tensor(torch.ones(1, 3)), valid),
        (Tensor(torch.ones(6)[::2]), valid), (valid, Tensor(torch.ones(6)[::2])),
        (negative, valid), (valid, negative)]
    large = Tensor(torch.empty(1 << 31, device="meta"))
    bad.append((large, large))
    aligned = Tensor(torch.ones(4))
    for remainder in (1, 2, 3):
        unaligned = Tensor(shared.unaligned_tensor(remainder))
        bad.extend([(unaligned, aligned), (aligned, unaligned)])
    for x, y in bad:
        calls.clear()
        try:
            module.add(x, y)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("unsupported input admitted")
        assert calls == [], "rejection must precede allocation"

    backing = torch.arange(20, dtype=torch.float32)
    good = [(torch.arange(n, dtype=torch.float32), torch.ones(n)) for n in (1, 1023, 1024, 1025, 2049)]
    good += [(torch.tensor(2.), torch.tensor(-3.)), (torch.empty(2, 0), torch.empty(2, 0)),
        (torch.as_strided(backing, (0,), (1,), 1000), torch.empty(0)),
        (shared.unaligned_tensor(1, empty=True), shared.unaligned_tensor(3, empty=True)),
        (torch.as_strided(backing, (1, 4), (99, 1), 2), torch.ones(1, 4)),
        (backing[2:7], backing[4:9]), (backing[1:6], backing[3:8]), (backing, backing),
        (torch.tensor([float("nan"), float("inf"), -float("inf")]), torch.ones(3))]
    for a, b in good:
        calls.clear()
        x, y = Tensor(a), Tensor(b)
        before_a, before_b = shared.raw_storage(a).clone(), shared.raw_storage(b).clone()
        out = module.add(x, y)
        torch.testing.assert_close(out.data, a + b, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(shared.raw_storage(a), before_a, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(shared.raw_storage(b), before_b, rtol=0, atol=0, equal_nan=True)
        assert out is not x and out is not y and out.shape == x.shape and out.data.is_contiguous()
        if x.numel():
            grid, actual_x, actual_y, actual_out, n, block = calls[1]
            assert len(calls) == 2 and actual_x is x and actual_y is y and actual_out is out
            assert (grid, n, block) == (((x.numel() + 1023) // 1024,), x.numel(), 1024)
        else:
            assert calls == ["allocation"]

    # The unchanged original body reads raw consecutive cells in these cases.
    for x in (torch.arange(10, dtype=torch.float32)[::2], torch._neg_view(torch.arange(1, 6, dtype=torch.float32))):
        y, out = torch.zeros_like(x), torch.empty_like(x)
        current_pid[0] = 0
        kernel(shared.Pointer(x), shared.Pointer(y), shared.Pointer(out), x.numel(), 1024)
        assert not torch.equal(out, x + y)
    assert not (ctypes.c_int32(0).value < ctypes.c_int32(1 << 31).value)
    print(f"Source binding, {len(bad)} rejected inputs, {len(good)} admitted cases and three original counterexamples passed.")
    print("Complete original input storage preserved; CPU interpreter and mocked CUDA host only.")


if __name__ == "__main__":
    source_binding()
    shared.check_tensor_semantics()
    host_and_counterexamples()
    shared.check_kernel_pair(1024)
