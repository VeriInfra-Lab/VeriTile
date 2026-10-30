"""Direct-buffer contract and actual kernel AST checks; CUDA is mocked."""
import ast
import contextlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "var_len_copy_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/var_len_copy/var_len_copy.py"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_binding():
    source = ast.parse(SOURCE.read_text())
    kernel = function(SOURCE, "var_len_copy_kernel_triton_repaired")
    masks = [ast.unparse(k.value) for n in ast.walk(kernel) if isinstance(n, ast.Call)
             for k in n.keywords if k.arg == "mask"]
    assert masks == ["i + old_offset < length", "i + new_offset < length"]
    casts = [ast.unparse(n) for n in ast.walk(kernel) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "tl.cast"]
    assert len(casts) == 6 and all(n.endswith(", tl.int64)") for n in casts)
    candidate = function(SOURCE, "var_len_copy_kernel_triton_commuted")
    original_load = next(n for n in ast.walk(kernel) if isinstance(n, ast.Call)
                         and ast.unparse(n.func) == "tl.load" and n.keywords)
    candidate_load = next(n for n in ast.walk(candidate) if isinstance(n, ast.Call)
                          and ast.unparse(n.func) == "tl.load" and n.keywords)
    assert ast.unparse(original_load.args[0]) == "old_a_location + old_start + i + old_offset"
    assert ast.unparse(candidate_load.args[0]) == "old_a_location + i + old_start + old_offset"
    candidate.name = kernel.name
    candidate_load.args[0] = original_load.args[0]
    assert ast.dump(candidate, include_attributes=False) == ast.dump(kernel, include_attributes=False)
    lean_source = (HERE / "Kernels.lean").read_text()
    assert "repairedVarLenCopy%[original]" in lean_source and "repairedVarLenCopy%[commuted]" in lean_source
    assert "mask=i + old_offset < length" in lean_source and "mask=i + new_offset < length" in lean_source
    assert "tl.region os = tl.int32, ol = tl.int32, ns = tl.int32, src = tl.float32, dst = tl.float32" in lean_source
    assert lean_source.count(".to(tl.int64)") == 6
    wrapper = function(SOURCE, "launch_var_len_copy_triton_repaired")
    forbidden = {"clone", "contiguous", "resolve_neg", "copy_", "to"}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in forbidden for n in ast.walk(wrapper))
    guards = [n for n in wrapper.body if isinstance(n, ast.If)
              and any(isinstance(x, ast.Raise) for x in n.body)]
    assert len(guards) == 15
    lean = (HERE / "Contract.lean").read_text()
    checks = lean.split("def checkInputs", 1)[1].split("def checkArguments", 1)[0]
    assert re.findall(r"\.error \.(\w+)", checks) == [
        "layout", "rank", "dtype", "cuda", "device", "contiguous", "negative",
        "count", "extent", "alignment", "overlap", "sourceBounds", "outputBounds", "destinations"]
    names = ["layout", "rank", "dtype", "cuda", "device", "contiguous", "noNeg", "count",
             "extent", "alignment", "separation", "sourceBound", "outputBound", "distinct"]
    contract = lean.split("structure InputContract", 1)[1].split("inductive InputError", 1)[0]
    assert all(f"  {name} :" in contract for name in names)
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    assert "FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert "Correct" not in (HERE / "FPEquiv.lean").read_text()


class Tensor:
    def __init__(self, data, device="cuda:0", cuda=True):
        self.data, self.device, self.is_cuda = data, torch.device(device), cuda
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self): return self.data.data_ptr()
    def cpu(self): return self.data.cpu()


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


def backing_bytes(t):
    return torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes(),), (1,))


class Pointer:
    def __init__(self, data, offset=0): self.data, self.offset = data, offset
    def __add__(self, offset): return Pointer(self.data, self.offset + offset)


def interpreter(path, name):
    pid = [0]
    def load(pointer, mask=None):
        storage = raw(pointer.data)
        offsets = pointer.offset + pointer.data.storage_offset()
        if isinstance(offsets, int):
            assert 0 <= offsets < len(storage)
            return int(storage[offsets])
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        selected = offsets[mask]
        assert bool(((selected >= 0) & (selected < len(storage))).all()), "source OOB"
        result = torch.zeros_like(offsets, dtype=pointer.data.dtype)
        result[mask] = storage[selected]
        return result
    def store(pointer, values, mask):
        storage = raw(pointer.data)
        indices = pointer.data.storage_offset() + pointer.offset[mask]
        assert bool(((indices >= 0) & (indices < len(storage))).all()), "output OOB"
        storage[indices] = values[mask]
    node = function(path, name)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    ns = {"tl": SimpleNamespace(program_id=lambda _: pid[0], arange=torch.arange,
        cast=lambda value, _: value.to(torch.int64) if isinstance(value, torch.Tensor) else int(value),
        int64=torch.int64, load=load, store=store)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), ns)
    return ns[name], pid


def main(kernel_name="var_len_copy_kernel_triton_repaired"):
    source_binding()
    spec = importlib.util.spec_from_file_location("direct_var_copy", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel, pid = interpreter(SOURCE, kernel_name)
    calls = []
    class Launch:
        def __getitem__(self, grid):
            def run(a, lengths, source, dest, output, block):
                calls.append((grid, a, lengths, source, dest, output, block))
                assert block == 256
                for p in range(grid[0]):
                    pid[0] = p
                    kernel(*(Pointer(t.data) for t in (a, lengths, source, dest, output)), block)
            return run
    module.var_len_copy_kernel_triton_repaired = Launch()
    module.torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, int32=torch.int32,
        float32=torch.float32, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    run = module.launch_var_len_copy_triton_repaired
    cases = [([], [], []), ([0], [0], [-7]), ([-9], [-1], [-7]),
             ([0], [1], [1]), ([0], [255], [2]), ([0], [256], [0]),
             ([0], [257], [3]), ([0], [511], [1]), ([0], [512], [0]),
             ([0], [513], [0]), ([0, 40, 3], [17, 31, 0], [100, 3, -1])]
    for starts, lengths, destinations in cases:
        bits = torch.arange(700, dtype=torch.int32) * 16381
        bits[:6] = torch.tensor([0, -2147483648, 2139095040, -8388608, 2143289345, 1], dtype=torch.int32)
        source = bits.view(torch.float32)
        output = torch.full((720,), 42., dtype=torch.float32)[3:703]
        args = [Tensor(torch.tensor(v, dtype=torch.int32)) for v in (starts, lengths)]
        args += [Tensor(source), Tensor(torch.tensor(destinations, dtype=torch.int32)), Tensor(output)]
        originals = [backing_bytes(t.data).clone() for t in args]
        expected = output.clone()
        for a, length, d in zip(starts, lengths, destinations):
            if length > 0: expected[d:d+length].copy_(source[a:a+length])
        calls.clear()
        assert run(*args) is None
        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
        for t, before in zip(args[:-1], originals[:-1]): assert torch.equal(backing_bytes(t.data), before)
        assert torch.equal(backing_bytes(output)[:12], originals[-1][:12])
        assert torch.equal(backing_bytes(output)[(3+700)*4:], originals[-1][(3+700)*4:])
        if starts:
            assert len(calls) == 1 and calls[0][1:6] == tuple(args)
        else:
            assert not calls
    def good():
        return [Tensor(torch.tensor([0, 2], dtype=torch.int32)),
                Tensor(torch.tensor([2, 2], dtype=torch.int32)), Tensor(torch.arange(8, dtype=torch.float32)),
                Tensor(torch.tensor([0, 4], dtype=torch.int32)), Tensor(torch.zeros(8))]
    bad = []
    a=good(); a[0]=[]; bad.append(a)
    a=good(); a[2]=Tensor(torch.ones(2, 4)); bad.append(a)
    a=good(); a[0]=Tensor(torch.tensor([0, 2], dtype=torch.int64)); bad.append(a)
    a=good(); a[2]=Tensor(torch.ones(8), cuda=False); bad.append(a)
    a=good(); a[2]=Tensor(torch.ones(8), device="cuda:1"); bad.append(a)
    a=good(); a[2]=Tensor(torch.ones(16)[::2]); bad.append(a)
    a=good(); a[2]=Tensor(torch._neg_view(torch.ones(8))); bad.append(a)
    a=good(); a[1]=Tensor(torch.tensor([2], dtype=torch.int32)); bad.append(a)
    a=good(); a[4]=a[2]; bad.append(a)
    a=good(); a[0]=Tensor(torch.tensor([-1, 0], dtype=torch.int32)); bad.append(a)
    a=good(); a[0]=Tensor(torch.tensor([7, 0], dtype=torch.int32)); bad.append(a)
    a=good(); a[3]=Tensor(torch.tensor([0, 7], dtype=torch.int32)); bad.append(a)
    a=good(); a[3]=Tensor(torch.tensor([0, 1], dtype=torch.int32)); bad.append(a)
    a=good(); a[2]=Tensor(torch.ones(8, dtype=torch.float64)); bad.append(a)
    for args in bad:
        calls.clear()
        try: run(*args)
        except (ValueError, TypeError): pass
        else: raise AssertionError("unsupported input accepted")
        assert not calls
    original, original_pid = interpreter(ORIGINAL, "var_len_copy_kernel_triton")
    source=torch.ones(513); output=torch.zeros(513)
    original(*(Pointer(t) for t in (torch.tensor([0], dtype=torch.int32),
        torch.tensor([257], dtype=torch.int32), source, torch.tensor([0], dtype=torch.int32), output)),256)
    assert output[257] == 1 and output[511] == 1 and output[512] == 0
    print(f"{kernel_name}: Source binding, {len(cases)} direct-copy cases, {len(bad)} rejection cases and original length-257 overcopy passed.")
    print("FP32 raw bits and caller backing bytes checked on CPU; CUDA context/launch mocked; no GPU claim.")


if __name__ == "__main__":
    main()
    main("var_len_copy_kernel_triton_commuted")
