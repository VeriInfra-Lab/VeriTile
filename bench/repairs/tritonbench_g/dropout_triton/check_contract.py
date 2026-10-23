"""Targeted AST, host admission and CPU source checks. This does not run CUDA."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dropout_triton_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/dropout_triton/dropout_triton.py"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_checks():
    assert ast.dump(function(SOURCE, "_dropout")) == ast.dump(function(ORIGINAL, "_dropout"))
    candidate = copy.deepcopy(function(SOURCE, "_dropout"))
    candidate.name = "_dropout_index_commuted"
    block_start = next(node for node in candidate.body
                       if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "block_start")
    block_start.value = ast.parse("BLOCK_SIZE * pid", mode="eval").body
    assert ast.dump(candidate) == ast.dump(function(SOURCE, "_dropout_index_commuted"))
    wrapper = function(SOURCE, "dropout_repaired")
    guards = [ast.unparse(n.test) for n in wrapper.body if isinstance(n, ast.If)]
    assert guards == [
        "not isinstance(x, torch.Tensor) or not isinstance(x_keep, torch.Tensor)",
        "x.layout != torch.strided or x_keep.layout != torch.strided",
        "x.dtype != torch.float32 or x_keep.dtype != torch.int32",
        "not (x.is_cuda and x_keep.is_cuda)", "x.device != x_keep.device",
        "x.shape != x_keep.shape", "not (x.is_contiguous() and x_keep.is_contiguous())",
        "x.is_neg() or x_keep.is_neg()", "n_elements >= 2 ** 31",
        "n_elements > 0 and (x.data_ptr() % 4 != 0 or x_keep.data_ptr() % 4 != 0)",
        "type(p) is not float", "not 0.0 <= p <= 1.0", "n_elements == 0"]
    context = next(n for n in wrapper.body if isinstance(n, ast.With))
    assert ast.unparse(context.items[0].context_expr) == "torch.cuda.device(x.device)"
    assert len(context.body) == 1 and isinstance(context.body[0], ast.Expr)
    assert ast.unparse(context.body[0]).startswith("_dropout[grid](")
    lean = (HERE / "Contract.lean").read_text()
    for text in ["y.dtype = .other \"int32\"", "sameShape : x.shape = y.shape",
                 "extent : x.numel < 2^31", "alignment : 0 < x.numel",
                 "if 0 ≤ q ∧ q ≤ 1", "| .other => .error .scalarType",
                 "| .float none => .error .probability"]:
        assert text in lean, text
    for entry in ["Correct.lean", "FPEquiv.lean"]:
        assert f"import bench.repairs.tritonbench_g.dropout_triton.{'FPEquiv' if entry == 'Correct.lean' else 'Correct'}" not in (HERE / entry).read_text()
    pending = [HERE / "Kernels.lean"]
    while pending:
        for line in pending.pop().read_text().splitlines():
            if line.startswith("import "):
                module = line.split()[1]
                assert ".Proofs." not in module
                if module.startswith("bench."):
                    pending.append(HERE.parents[3] / (module.replace(".", "/") + ".lean"))


class Tensor:
    def __init__(self, data, *, device="cuda:0", cuda=True, count=None):
        self.data, self.device, self.is_cuda, self.count = data, device, cuda, count
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    @property
    def shape(self): return self.data.shape
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel() if self.count is None else self.count
    def data_ptr(self):
        assert self.numel() > 0, "empty input pointer must not be queried"
        return self.data.data_ptr()


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


class Pointer:
    def __init__(self, tensor, offsets=None): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, offsets)


def interpreter():
    pid, calls, current_device, allocated = [0], [], ["cuda:0"], []
    def load(pointer, mask):
        t = pointer.tensor
        # Inactive values are arbitrary. They cannot affect any active store.
        values = torch.full_like(pointer.offsets, 17, dtype=t.dtype)
        values[mask] = raw(t)[t.storage_offset() + pointer.offsets[mask]]
        return values
    def store(pointer, values, mask):
        t = pointer.tensor
        raw(t)[t.storage_offset() + pointer.offsets[mask]] = values[mask]
    node = function(SOURCE, "_dropout")
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    namespace = {"tl": SimpleNamespace(program_id=lambda axis: pid[0], arange=torch.arange,
                 load=load, store=store, where=lambda k, x, z: torch.where(k != 0, x, z))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)
    source_kernel = namespace["_dropout"]
    class Launch:
        def __getitem__(self, grid):
            def run(x, keep, out, n, p, BLOCK_SIZE):
                assert current_device[0] == x.device == keep.device == out.device
                assert out is allocated[-1], "launch must use the allocated object"
                calls.append("launch")
                shape = grid({"BLOCK_SIZE": BLOCK_SIZE})
                assert shape == ((n + 1023) // 1024,)
                for block in range(shape[0]):
                    pid[0] = block
                    source_kernel(Pointer(x.data), Pointer(keep.data), Pointer(out.data),
                        n, torch.tensor(p, dtype=torch.float32), BLOCK_SIZE)
            return run
    def allocate(x):
        calls.append("allocation")
        result = Tensor(torch.empty_like(x.data), device=x.device)
        allocated.append(result)
        return result
    class DeviceContext:
        def __init__(self, device): self.device = device
        def __enter__(self):
            self.previous = current_device[0]
            current_device[0] = self.device
            calls.append("enter " + self.device)
        def __exit__(self, *exc):
            current_device[0] = self.previous
            calls.append("exit " + self.device)
    namespace.update(torch=SimpleNamespace(Tensor=Tensor, strided=torch.strided,
        float32=torch.float32, int32=torch.int32, empty_like=allocate,
        cuda=SimpleNamespace(device=DeviceContext)),
        triton=SimpleNamespace(cdiv=lambda n, b: (n+b-1)//b), _dropout=Launch())
    candidate = copy.deepcopy(function(SOURCE, "_dropout"))
    candidate.name = "_dropout_index_commuted"
    block_start = next(node for node in candidate.body
                       if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "block_start")
    block_start.value = ast.parse("BLOCK_SIZE * pid", mode="eval").body
    assert ast.dump(candidate) == ast.dump(function(SOURCE, "_dropout_index_commuted"))
    wrapper = function(SOURCE, "dropout_repaired")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace["dropout_repaired"], calls, current_device, allocated


def unaligned(dtype, offset, empty=False):
    storage = torch.arange(30, dtype=torch.uint8).untyped_storage()[offset:offset+16]
    return torch.empty(0, dtype=dtype).set_(storage, 0, (0 if empty else 4,), (1,))


def checks():
    dropout, calls, current_device, allocated = interpreter()
    good = Tensor(torch.arange(4, dtype=torch.float32))
    keep = Tensor(torch.tensor([0, 1, 2, -1], dtype=torch.int32))
    rejected = [
        (None, keep, .5), (good, None, .5), (Tensor(good.data, cuda=False), keep, .5),
        (good, Tensor(keep.data, device="cuda:1"), .5),
        (Tensor(good.data.half()), keep, .5), (good, Tensor(keep.data.bool()), .5),
        (good, Tensor(keep.data[:3]), .5), (good, Tensor(keep.data.reshape(2, 2)), .5),
        (Tensor(torch.arange(8.).reshape(2, 4).T), Tensor(torch.ones((4, 2), dtype=torch.int32)), .5),
        (good, Tensor(torch.arange(8, dtype=torch.int32)[::2]), .5),
        (Tensor(torch._neg_view(good.data)), keep, .5),
        (good, Tensor(torch._neg_view(keep.data)), .5),
        (Tensor(good.data, count=2**31), keep, .5),
        (good, keep, 1), (good, keep, True), (good, keep, float("nan")),
        (good, keep, float("inf")), (good, keep, -.1), (good, keep, 1.1),
    ]
    for offset in (1, 2, 3):
        rejected += [(Tensor(unaligned(torch.float32, offset)), keep, .5),
                     (good, Tensor(unaligned(torch.int32, offset)), .5)]
    for x, k, p in rejected:
        calls.clear()
        try: dropout(x, k, p)
        except (ValueError, TypeError): pass
        else: raise AssertionError("unsupported input accepted")
        assert calls == [], "rejection must precede allocation"
    valid = []
    for n in (0, 1, 1023, 1024, 1025, 2057):
        values = torch.arange(n, dtype=torch.float32) - 8
        mask = (torch.arange(n, dtype=torch.int32) % 3) - 1
        valid.append((Tensor(values), Tensor(mask), .5))
    valid += [(good, keep, 0.0), (good, keep, -0.0), (good, keep, 1.0), (good, Tensor(torch.zeros_like(keep.data)), 1.0), (good, keep, 1.0 - 2**-26),
              (Tensor(torch.tensor([float("nan"), float("inf"), -float("inf"), -0.])), keep, .5),
              (Tensor(torch.arange(12.)[3:7]), Tensor(torch.arange(12, dtype=torch.int32)[5:9]), .25),
              (Tensor(torch.tensor(-2.)), Tensor(torch.tensor(-1, dtype=torch.int32)), .5),
              (Tensor(unaligned(torch.float32, 1, True)), Tensor(unaligned(torch.int32, 3, True)), 1.0),
              (good, Tensor(good.data.view(torch.int32)), .5),
              (Tensor(good.data, device="cuda:1"), Tensor(keep.data, device="cuda:1"), .5),
              (Tensor(torch.empty(0), device="cuda:1"), Tensor(torch.empty(0, dtype=torch.int32), device="cuda:1"), .5)]
    # Distinct contiguous views may overlap while interpreting the same words
    # as FP32 data and signed-int32 keep values. Admission imposes no separation.
    alias_backing = torch.arange(8, dtype=torch.float32) + 1
    valid.append((Tensor(alias_backing[1:5]),
                  Tensor(alias_backing.view(torch.int32)[3:7]), .5))
    for x, k, p in valid:
        calls.clear()
        before_x, before_k = raw(x.data).clone(), raw(k.data).clone()
        result = dropout(x, k, p)
        assert result is allocated[-1], "return must retain the allocated object identity"
        assert current_device[0] == "cuda:0", "device context must restore the caller device"
        p32 = torch.tensor(p, dtype=torch.float32)
        expected = torch.where(k.data != 0, x.data / (1 - p32), 0.)
        torch.testing.assert_close(result.data, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(raw(x.data), before_x, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(raw(k.data), before_k, rtol=0, atol=0)
        assert calls == (["allocation"] if x.numel() == 0 else
                         ["allocation", "enter " + x.device, "launch", "exit " + x.device])
        assert result.data.shape == x.data.shape and result.data.dtype == torch.float32
    # Concrete original-domain defects, reproduced with real CPU views.
    physical_mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    strided_mask = physical_mask[::2]
    assert strided_mask.tolist() == [1, 1] and raw(strided_mask)[:2].tolist() == [1, 0]
    neg = torch._neg_view(torch.tensor([2.]))
    assert neg.is_contiguous() and neg.item() == -2. and raw(neg).item() == 2.
    assert torch.tensor(1.0 - 2**-26, dtype=torch.float32).item() == 1.0
    print(f"Passed: unchanged AST, source/contract binding, {len(rejected)} rejections, {len(valid)} admitted CPU source executions.")
    print("CUDA allocation/launch were mocked; no GPU or native numerical conformance is claimed.")


if __name__ == "__main__":
    source_checks()
    checks()
