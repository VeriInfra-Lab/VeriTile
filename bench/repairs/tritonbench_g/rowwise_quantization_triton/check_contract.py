"""Targeted actual-source CPU AST and admission tests; native numerics deferred."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "rowwise_quantization_triton_repaired.py"
spec = importlib.util.spec_from_file_location("add_contract_helpers", HERE.parent / "add_example/check_contract.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

class Tensor(helpers.TensorFacade):
    @property
    def ndim(self): return self.data.ndim


def function(name):
    return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    pid = [0]
    calls = []
    class Pointer:
        def __init__(self, tensor, offsets=0): self.tensor, self.offsets = tensor, offsets
        def __add__(self, offsets): return Pointer(self.tensor, offsets)

    def load(ptr, mask):
        offsets = ptr.offsets.long()
        result = torch.zeros_like(offsets, dtype=torch.float32)
        result[mask] = helpers.raw_storage(ptr.tensor)[ptr.tensor.storage_offset() + offsets[mask]]
        return result

    def store(ptr, value, mask=None):
        target = ptr.tensor.reshape(-1)
        if mask is None:
            target[ptr.offsets.long()] = value.to(target.dtype)
        else:
            target[ptr.offsets[mask].long()] = value[mask].to(target.dtype)

    tl = SimpleNamespace(int64=torch.int64, program_id=lambda axis: torch.tensor(pid[0]),
                         full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
                         arange=lambda a, b: torch.arange(a, b, dtype=torch.int32),
                         load=load, store=store, abs=torch.abs, where=torch.where,
                         max=lambda x, axis: torch.max(x),
                         extra=SimpleNamespace(cuda=SimpleNamespace(libdevice=SimpleNamespace(llrint=lambda x: torch.round(x).to(torch.int64)))))
    node = function("_quantize_rowwise")
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    kernel_ns = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), kernel_ns)

    class Launch:
        def __getitem__(self, grid):
            def run(x, q, m, n, BLOCK_SIZE, P2):
                calls.append("launch")
                for row in reversed(range(x.shape[0])):
                    pid[0] = row
                    kernel_ns["_quantize_rowwise"](Pointer(x.data), Pointer(q.data), Pointer(m.data), n, BLOCK_SIZE, P2)
            return run

    def allocate(shape, device, dtype):
        calls.append("allocation")
        return Tensor(torch.empty(shape, dtype=dtype), str(device))

    fake_torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
        int8=torch.int8, float16=torch.float16, empty=allocate,
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    ns = {"torch": fake_torch, "triton": SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
          "_quantize_rowwise": Launch()}
    nodes = [function("_check"), function("quantize_rowwise")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), ns)

    # Admission syntax ties the Python checks to the named Lean contract.
    guards = [ast.unparse(n.test) for n in function("_check").body if isinstance(n, ast.If)]
    expected = ["not isinstance(x, torch.Tensor)", "x.layout != torch.strided or x.dtype != torch.float32",
                "not x.is_cuda", "x.ndim != 2", "not x.is_contiguous() or x.is_neg()",
                "not 0 < x.shape[1] <= 2 ** 20", "x.shape[0] >= 2 ** 31", "x.numel() > 0 and x.data_ptr() % 4 != 0"]
    assert guards == expected
    assignments = {n.targets[0].id: ast.unparse(n.value) for n in function("_quantize_rowwise").body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert assignments["pid"] == "tl.program_id(axis=0).to(tl.int64)"
    assert assignments["block_start"] == "pid * tl.full((), BLOCK_SIZE, tl.int64)"
    assert assignments["offsets"] == "block_start + arange"
    assert assignments["row_mask"] == "arange < BLOCK_SIZE"
    assert assignments["max_val"] == "tl.max(tl.where(row_mask, abs_x, 0), axis=0)"
    assert assignments["safe_max"] == "tl.where(max_val == 0.0, 1.0, max_val)"
    assert assignments["output"] == "tl.extra.cuda.libdevice.llrint(127.0 * (x / safe_max))"
    wrapper = function("quantize_rowwise")
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("clone", "resolve_neg", "contiguous", "copy_") for n in ast.walk(wrapper))
    contract = (HERE / "Contract.lean").read_text()
    for field in ("layout", "dtype", "cuda", "rank", "contiguous", "noNeg", "width", "grid", "alignment"):
        assert f"  {field} :" in contract
    assert "columns x≤2^20" in contract and "rows x<2^31" in contract
    source = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in source

    accepted = 0
    for rows, cols in [(0, 1), (0, 2**20), (1, 1), (2, 3), (3, 17), (2, 129), (1, 513)]:
        for zero in (False, True):
            backing = torch.zeros(rows * cols + 6) if zero else torch.arange(rows * cols + 6, dtype=torch.float32) - 10
            data = backing[2:2 + rows * cols].reshape(rows, cols)
            before = backing.clone()
            x = Tensor(data)
            calls.clear()
            q, m = ns["quantize_rowwise"](x)
            maximum = data.abs().amax(dim=1)
            safe = torch.where(maximum == 0, 1, maximum)
            expected_q = torch.round(127 * (data / safe[:, None])).to(torch.int8)
            torch.testing.assert_close(q.data, expected_q, rtol=0, atol=0)
            torch.testing.assert_close(m.data, maximum.to(torch.float16), rtol=0, atol=0)
            torch.testing.assert_close(backing, before, rtol=0, atol=0)
            assert q.dtype == torch.int8 and m.dtype == torch.float16
            assert q.shape == x.shape and m.shape == (rows,)
            assert calls == ["allocation", "allocation"] + (["launch"] if rows else [])
            accepted += 1

    bad = [[], Tensor(torch.eye(2).to_sparse()), Tensor(torch.ones(2, 2, dtype=torch.float64)),
           Tensor(torch.ones(2, 2), "cpu", False), Tensor(torch.ones(3)),
           Tensor(torch.ones(2, 3).T), Tensor(torch._neg_view(torch.ones(2, 3))),
           Tensor(torch.empty(2, 0)), Tensor(torch.empty(0, 2**20 + 1)),
           Tensor(torch.empty(2**31, 1, device="meta"))]
    bad.extend(Tensor(helpers.unaligned_tensor(k).reshape(1, 4)) for k in (1, 2, 3))
    for x in bad:
        calls.clear()
        try: ns["quantize_rowwise"](x)
        except (TypeError, ValueError): pass
        else: raise AssertionError("unsupported input admitted")
        assert calls == []
    print(f"PASS: {accepted} complete CPU AST output-pair/frame cases; {len(bad)} guard rejections; width/grid limits; zero rows and empty batch.")
    print("CUDA and launches are mocked; this is a CPU AST check, not a GPU execution or numerical-assumption experiment.")

if __name__ == "__main__": main()
