"""Targeted source/guard and CPU backing-storage regressions; no GPU claim."""
import ast
import copy
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "triton_mul2_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/triton_mul2/triton_mul2.py"
# Reuse the existing contract example's real Tensor metadata and storage helpers.
spec = importlib.util.spec_from_file_location("add_contract_helpers", HERE.parent / "add_example/check_contract.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
TensorFacade, Pointer = helpers.TensorFacade, helpers.Pointer
BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_binding():
    for name in ("mul2_kernel", "mul2_inplace_kernel"):
        assert ast.dump(function(SOURCE, name)) == ast.dump(function(ORIGINAL, name))
    for original, candidate in (("mul2_kernel", "mul2_commuted_kernel"),
                                ("mul2_inplace_kernel", "mul2_inplace_commuted_kernel")):
        expected_candidate = copy.deepcopy(function(SOURCE, original))
        expected_candidate.name = candidate
        assignment = next(node for node in expected_candidate.body
                          if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "output")
        assignment.value = ast.parse("x * 2", mode="eval").body
        assert ast.dump(function(SOURCE, candidate)) == ast.dump(expected_candidate)
    expected = [
        "not isinstance(x, torch.Tensor)", "x.layout != torch.strided", "x.dtype != torch.float32",
        "not x.is_cuda", "not x.is_contiguous()", "x.is_neg()", "x.numel() >= 2**31",
        "x.numel() > 0 and x.data_ptr() % 4 != 0",
        "type(BLOCK_SIZE) is not int or BLOCK_SIZE not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)",
    ]
    guards = [n.test for n in function(SOURCE, "_check").body if isinstance(n, ast.If)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(e, mode="eval").body) for e in expected]
    contract = (HERE / "Contract.lean").read_text()
    for field in ("layout", "dtype", "cuda", "contiguous", "noNeg", "extent", "alignment", "block"):
        assert f"  {field} :" in contract
    assert "[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576]" in contract
    assert "block ∉ supportedBlocks" in contract
    assert "x.numel >= 2^31" in contract and "x.numel > 0 && x.dataPtr.toNat % 4 != 0" in contract
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    for name in ("triton_mul2", "triton_mul2_inplace"):
        wrapper = function(SOURCE, name)
        assert ast.unparse(wrapper.body[0]) == "_check(x, BLOCK_SIZE)"
        assert not any(isinstance(n, ast.Attribute) and n.attr in ("clone", "copy_", "resolve_neg", "contiguous")
                       for n in ast.walk(wrapper))


def kernels():
    pid = [0]

    def load(pointer, mask):
        values = torch.zeros_like(pointer.offsets, dtype=torch.float32)
        values[mask] = helpers.raw_storage(pointer.tensor)[pointer.tensor.storage_offset() + pointer.offsets[mask]]
        return values

    def store(pointer, values, mask):
        helpers.raw_storage(pointer.tensor)[pointer.tensor.storage_offset() + pointer.offsets[mask]] = values[mask]

    env = {"tl": SimpleNamespace(program_id=lambda axis: pid[0], arange=torch.arange, load=load, store=store)}
    nodes = []
    for name in ("mul2_kernel", "mul2_inplace_kernel"):
        node = function(SOURCE, name)
        node.decorator_list = []
        for argument in node.args.args:
            argument.annotation = None
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), env)
    return env, pid


def main():
    source_binding()
    env, pid = kernels()
    calls = []

    def allocate(x):
        calls.append("allocate")
        return TensorFacade(torch.zeros_like(x.data), str(x.device))

    class Launch:
        def __init__(self, inplace): self.inplace = inplace
        def __getitem__(self, grid):
            def run(*args):
                calls.append("launch")
                if self.inplace:
                    x, n, block = args
                    out = x
                else:
                    x, out, n, block = args
                # Reverse scheduling exercises independent in-place programs.
                for program in reversed(range(grid[0])):
                    pid[0] = program
                    if self.inplace:
                        env["mul2_inplace_kernel"](Pointer(x.data), n, block)
                    else:
                        env["mul2_kernel"](Pointer(x.data), Pointer(out.data), n, block)
            return run

    namespace = {
        "torch": SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
                                 zeros_like=allocate, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext())),
        "triton": SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b),
        "mul2_kernel": Launch(False), "mul2_inplace_kernel": Launch(True),
    }
    nodes = [function(SOURCE, name) for name in ("_check", "triton_mul2", "triton_mul2_inplace")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    cases = 0
    for block in BLOCKS:
        for shape in ((0,), (), (1,), (17,), (3, 5), (2, 0, 3)):
            for inplace in (False, True):
                count = torch.Size(shape).numel()
                backing = torch.arange(count + 8, dtype=torch.float32) - 3
                x = TensorFacade(backing[3:3 + count].reshape(shape))
                before = backing.clone()
                expected = 2 * x.data.clone()
                calls.clear()
                out = namespace["triton_mul2_inplace" if inplace else "triton_mul2"](x, block)
                torch.testing.assert_close(out.data, expected, rtol=0, atol=0)
                assert out.device == x.device and out.shape == x.shape and out.dtype == x.dtype
                if inplace:
                    assert out is x
                    torch.testing.assert_close(backing[:3], before[:3], rtol=0, atol=0)
                    torch.testing.assert_close(backing[3 + count:], before[3 + count:], rtol=0, atol=0)
                else:
                    assert out is not x and out.data.untyped_storage().data_ptr() != x.data.untyped_storage().data_ptr()
                    torch.testing.assert_close(backing, before, rtol=0, atol=0)
                assert calls.count("launch") == int(count > 0)
                assert calls.count("allocate") == int(not inplace)
                cases += 1

    bad = [[], TensorFacade(torch.ones(4)[::2]), TensorFacade(torch._neg_view(torch.ones(3))),
           TensorFacade(torch.ones(3, dtype=torch.float64)), TensorFacade(torch.eye(3).to_sparse()),
           TensorFacade(torch.ones(3), "cpu", False), TensorFacade(torch.empty(2**31, device="meta"))]
    bad.extend(TensorFacade(helpers.unaligned_tensor(k)) for k in (1, 2, 3))
    rejects = 0
    for inplace in (False, True):
        wrapper = namespace["triton_mul2_inplace" if inplace else "triton_mul2"]
        for x in bad:
            calls.clear()
            try: wrapper(x)
            except (TypeError, ValueError): pass
            else: raise AssertionError("unsupported tensor admitted")
            assert calls == []
            rejects += 1
        for block in (-1, 0, 3, 2**21, 16.0, True, "16"):
            calls.clear()
            try: wrapper(TensorFacade(torch.ones(3)), block)
            except ValueError: pass
            else: raise AssertionError("unsupported block admitted")
            assert calls == []
            rejects += 1
        empty = TensorFacade(helpers.unaligned_tensor(1, empty=True))
        wrapper(empty, 2**20)
        boundary = TensorFacade(torch.arange(9, dtype=torch.float32))
        torch.testing.assert_close(wrapper(boundary, 4096).data, 2 * torch.arange(9, dtype=torch.float32), rtol=0, atol=0)
        strange = TensorFacade(torch.as_strided(torch.arange(8, dtype=torch.float32), (1, 3), (99, 1), 2))
        expected = 2 * strange.data.clone()
        torch.testing.assert_close(wrapper(strange).data, expected, rtol=0, atol=0)
        special = TensorFacade(torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0]))
        wrapper(special)

    # Original flat-pointer bug, directly executing the unchanged kernel AST.
    backing = torch.arange(8, dtype=torch.float32) + 1
    sliced = backing[::2]
    before = backing.clone()
    pid[0] = 0
    env["mul2_inplace_kernel"](Pointer(sliced), sliced.numel(), 16)
    assert backing[1] != before[1]  # A caller cell outside the logical view changed.
    assert not torch.equal(sliced, 2 * before[::2])
    print(f"PASS: unchanged source ASTs; {cases} whole-wrapper CPU AST cases; {rejects} guard rejections; empty/singleton/NaN/Inf admission; original stride counterexample.")
    print("CPU backing-storage and mocked CUDA launch only; no GPU or independent numerical replay claim. FP32 MUL-COMMUTE uses the existing report.")


if __name__ == "__main__":
    main()
