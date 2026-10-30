"""Actual wrapper and JIT AST replay on CPU storage; CUDA is mocked."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "swiglu_fwd_checked.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/swiglu_fwd/swiglu_fwd.py"
BLOCKS = (32, 64, 128, 256, 512, 1024)


def function(path, name):
    return next(x for x in ast.parse(path.read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == name)


class Tensor:
    def __init__(self, data, device="cuda:1", cuda=True):
        self.data, self.device, self.is_cuda = data, torch.device(device), cuda
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    def stride(self, axis): return self.data.stride(axis)
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.data.numel(), "empty pointers must not be queried"
        return self.data.data_ptr()
    def storage_offset(self): return self.data.storage_offset()
    def untyped_storage(self): return self.data.untyped_storage()
    def wrap(self, value): return Tensor(value, self.device, self.is_cuda)
    def contiguous(self): return self.wrap(self.data.contiguous())
    def reshape(self, *shape): return self.wrap(self.data.reshape(*shape))
    def chunk(self, *args, **kwargs): return tuple(self.wrap(x) for x in self.data.chunk(*args, **kwargs))


def storage(t):
    return torch.empty(0, dtype=torch.float32).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // 4,), (1,))


class Pointer:
    def __init__(self, tensor, offset=0): self.tensor, self.offset = tensor, offset
    def __add__(self, offset): return Pointer(self.tensor, self.offset + offset)


def cpu_kernel(path, name):
    ids = [0, 0]
    def load(ptr, mask, other):
        out = torch.full(mask.shape, other, dtype=torch.float32)
        out[mask] = storage(ptr.tensor)[ptr.tensor.storage_offset() + ptr.offset[mask]]
        return out
    def store(ptr, value, mask):
        storage(ptr.tensor)[ptr.tensor.storage_offset() + ptr.offset[mask]] = value[mask]
    fn = function(path, name)
    fn.decorator_list = []
    for a in fn.args.args: a.annotation = None
    ns = {"tl": SimpleNamespace(program_id=lambda a: ids[a], arange=torch.arange,
          load=load, store=store, float32=torch.float32, sigmoid=torch.sigmoid)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), ns)
    return ns[name], ids


def load_host(path=SOURCE, name="swiglu_fwd_checked", kernel_name="_swiglu_fwd_kernel",
              blocks=BLOCKS, historical_overlap=False):
    kernel, ids = cpu_kernel(path, kernel_name)
    events = []
    @contextlib.contextmanager
    def device(d):
        events.append(("device", d))
        yield
    def empty_like(t):
        events.append(("allocation", tuple(t.shape)))
        return t.wrap(torch.empty_like(t.data))
    class Launch:
        def __getitem__(self, grid):
            def run(x, y, out, sx, sy, so, n):
                for block in blocks:
                    actual_grid = grid({"BLOCK_N": block})
                    events.append(("launch", block, actual_grid, sx, sy, so, n))
                    for row in reversed(range(actual_grid[0])):
                        ids[0] = row
                        for col in reversed(range(actual_grid[1])):
                            ids[1] = col
                            kernel(Pointer(x.data), Pointer(y.data), Pointer(out.data), sx, sy, so, n, block)
            return run
    ns = {"torch": SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
          empty_like=empty_like, cuda=SimpleNamespace(device=device)),
          "triton": SimpleNamespace(cdiv=lambda n,b: (n+b-1)//b), kernel_name: Launch()}
    functions = [function(path, "_check_payload"), function(path, name)]
    if historical_overlap:
        # Exact previous repair guard, retained only to replay its admitted defect.
        guard = next(s for s in functions[-1].body
                     if isinstance(s, ast.If) and ast.unparse(s.test) == "M > 0")
        guard.body[1:] = ast.parse("""
same_storage = x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
x_begin, out_begin = x.storage_offset(), out.storage_offset()
x_end = x_begin + (M - 1) * sx + 2 * N
out_end = out_begin + (M - 1) * so + N
if same_storage and not (out_end <= x_begin or x_end <= out_begin):
    raise ValueError("output span overlaps the live input span")
""").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), ns)
    return ns[name], events


def check_source():
    assert ast.dump(function(SOURCE, "_swiglu_fwd_kernel")) == ast.dump(function(ORIGINAL, "_swiglu_fwd_kernel"))
    lean = (HERE / "Contract.lean").read_text()
    checks = ["positive", "sameDevice", "rows32", "inputStride32", "outputStride32",
              "inputPadded32", "outputPadded32", "unique", "alignment", "separate"]
    for field in checks: assert f"  {field} :" in lean
    payload = [ast.unparse(n.test) for n in function(SOURCE, "_check_payload").body if isinstance(n, ast.If)]
    assert payload == ["not isinstance(tensor, torch.Tensor)",
        "tensor.layout != torch.strided or tensor.dtype != torch.float32", "not tensor.is_cuda", "tensor.is_neg()"]
    body = function(SOURCE, "swiglu_fwd_checked")
    direct = [ast.unparse(n.test) for n in body.body if isinstance(n, ast.If)]
    expected = ["out is not None", "xy.ndim == 0 or xy.shape[-1] == 0 or xy.shape[-1] % 2 != 0",
        "xy.stride(-1) != 1", "out is None", "out.device != x.device",
        "not (M < 2 ** 31 and sx < 2 ** 31 and so < 2 ** 31 and (M * sx + N + 1024 < 2 ** 31) and (M * so + N + 1024 < 2 ** 31))",
        "M > 1 and so < N", "M > 0"]
    assert [ast.dump(ast.parse(x, mode="eval")) for x in direct] == [ast.dump(ast.parse(x, mode="eval")) for x in expected]
    original = function(ORIGINAL, "_swiglu_fwd")
    for stmt in original.body:
        assert any(ast.dump(stmt) == ast.dump(s) for s in body.body), ast.unparse(stmt)
    assert '.Proofs.' not in (HERE / "Kernels.lean").read_text()


def check_host():
    forward, events = load_host()
    torch.manual_seed(37)
    cases = [torch.randn(3, 66), torch.randn(2, 3, 2050), torch.randn(3, 130)[:, ::2].repeat_interleave(2, -1),
             torch.randn(3, 132)[:, ::2], torch.randn(2, 3, 66).transpose(0,1),
             torch.randn(2, 160)[:, 5:135], torch.empty(0, 66), torch.empty(2, 0, 66), torch.randn(66)]
    for i, data in enumerate(cases):
        x, before = Tensor(data), data.clone()
        expected = (data[..., :data.shape[-1]//2] * torch.sigmoid(data[..., :data.shape[-1]//2])
                    * data[..., data.shape[-1]//2:])
        supplied = None
        backing = None
        if i in (0, 1, 5):
            rows, cols = expected.numel()//expected.shape[-1], expected.shape[-1]
            backing = torch.full((rows*(cols+7)+16,), -99.)
            output = torch.as_strided(backing, (rows, cols), (cols+7, 1), 4)
            supplied = Tensor(output.reshape(expected.shape))
        events.clear()
        result = forward(x, supplied)
        # CPU vector sigmoid can differ by a few ulps with vector width.
        torch.testing.assert_close(result.data, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(data, before, rtol=0, atol=0)
        assert result.shape == expected.shape and result.device == x.device
        launches = [e for e in events if e[0] == "launch"]
        assert [e[1] for e in launches] == list(BLOCKS)
        assert ("device", x.device.index) in events
        if backing is not None:
            assert result.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
            written = set()
            for r in range(rows):
                for q in range(cols): written.add(4+r*(cols+7)+q)
            assert all(backing[k] == -99 for k in range(backing.numel()) if k not in written)
    failures = [([], None, TypeError), (Tensor(torch.ones(2, 4, dtype=torch.float64)), None, TypeError),
        (Tensor(torch.ones(2,4), "cpu", False), None, ValueError),
        (Tensor(torch._neg_view(torch.ones(2,4))), None, ValueError),
        (Tensor(torch.tensor(1.)), None, ValueError), (Tensor(torch.ones(2,3)), None, ValueError),
        (Tensor(torch.empty(2,0)), None, ValueError),
        (Tensor(torch.ones(2,4)), Tensor(torch.ones(2,2), "cuda:0"), ValueError),
        (Tensor(torch.ones(2,4)), Tensor(torch.ones(2,3)), AssertionError),
        (Tensor(torch.ones(2,4)), Tensor(torch.ones(2,4)[:, ::2]), AssertionError),
        (Tensor(torch.ones(2,4)), Tensor(torch.ones(2).expand(2,2)), ValueError)]
    overlap = torch.randn(3, 8)
    failures.append((Tensor(overlap), Tensor(overlap[:, :4]), ValueError))
    failures.append((Tensor(torch.empty(1 << 30, 2, device="meta")),
                     Tensor(torch.empty(1 << 30, 1, device="meta")), ValueError))
    for offset in (1, 2, 3):
        bytes_ = torch.arange(32, dtype=torch.uint8).untyped_storage()[offset:offset+16]
        unaligned = torch.empty(0).set_(bytes_, 0, (1,4), (4,1))
        failures.append((Tensor(unaligned), None, ValueError))
    for x, out, error in failures:
        events.clear()
        try: forward(x, out)
        except error: pass
        else: raise AssertionError("unsupported input admitted")
        assert not any(e[0] == "launch" for e in events)
    print(f"Source AST unchanged; {len(cases)} host paths × six configurations and {len(failures)} rejections passed.")
    print(f"CPU storage execution only; CUDA mocked. CUDA available: {torch.cuda.is_available()}.")

def check_original_counterexamples():
    # An odd final dimension leaves Y one element shorter than the kernel mask.
    kernel, ids = cpu_kernel(ORIGINAL, "_swiglu_fwd_kernel")
    xy = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    x, y = xy.chunk(2, dim=-1)
    try:
        kernel(Pointer(x), Pointer(y), Pointer(torch.empty(1, 2)), 3, 3, 2, 2, 32)
    except IndexError:
        pass
    else:
        raise AssertionError("original odd-width out-of-bounds counterexample disappeared")
    assert (torch.tensor(1 << 30, dtype=torch.int32) * 2).item() == -(1 << 31)
    print("Original counterexamples retained: odd-width Y overread and signed32 row product overflow.")


def check_storage_alias_regression():
    def overlapping_views():
        xy = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        out = torch.empty(0, dtype=torch.float32).set_(xy.untyped_storage()[4:], 0, (2, 4), (4, 1))
        assert xy.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
        assert out.data_ptr() == xy.data_ptr() + 4
        return xy, out

    previous, events = load_host(blocks=(32,), historical_overlap=True)
    xy, out = overlapping_views()
    expected = xy[:, :4] * torch.sigmoid(xy[:, :4]) * xy[:, 4:]
    actual = previous(Tensor(xy), Tensor(out)).data
    assert any(e[0] == "launch" for e in events)
    assert not torch.allclose(actual, expected)
    torch.testing.assert_close(actual[0], torch.tensor(
        [3.6552929878, 206.0810699463, 400.0629882813, 648.1182861328]), rtol=1e-6, atol=1e-6)

    current, events = load_host()
    for reverse in (False, True):
        xy, out = overlapping_views()
        if reverse:
            backing = torch.arange(1, 25, dtype=torch.float32)
            xy = torch.empty(0).set_(backing.untyped_storage()[4:], 0, (2, 8), (8, 1))
            out = backing[:8].reshape(2, 4)
        before = storage(xy).clone()
        events.clear()
        try:
            current(Tensor(xy), Tensor(out))
        except ValueError as error:
            assert "overlaps the live input span" in str(error)
        else:
            raise AssertionError("distinct-base overlapping storage was admitted")
        assert not any(e[0] == "launch" for e in events)
        torch.testing.assert_close(storage(xy), before, rtol=0, atol=0)

    # Whole storage objects overlap in these cases; their active byte spans do not.
    # Both touching orders, same storage and distinct storage objects stay admitted.
    for distinct in (False, True):
        for output_first in (False, True):
            backing = torch.arange(1, 33, dtype=torch.float32)
            xy = backing[8:24].reshape(2, 8) if output_first else backing[:16].reshape(2, 8)
            output_start = 0 if output_first else 16
            if distinct:
                if output_first:
                    xy = torch.empty(0).set_(backing.untyped_storage()[32:], 0, (2, 8), (8, 1))
                    out = backing[:8].reshape(2, 4)
                else:
                    out = torch.empty(0).set_(backing.untyped_storage()[4:], 15, (2, 4), (4, 1))
                assert xy.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
            else:
                out = backing[output_start:output_start+8].reshape(2, 4)
            before = backing.clone()
            expected = xy[:, :4] * torch.sigmoid(xy[:, :4]) * xy[:, 4:]
            events.clear()
            actual = current(Tensor(xy), Tensor(out)).data
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            assert len([e for e in events if e[0] == "launch"]) == len(BLOCKS)
            outside = torch.ones(backing.numel(), dtype=torch.bool)
            outside[output_start:output_start+8] = False
            torch.testing.assert_close(backing[outside], before[outside], rtol=0, atol=0)
    print("Previous repair alias corruption reproduced; current guard rejects both overlap orders.")
    print("Four touching/disjoint active-span cases pass all six configurations and shared-storage frame checks.")


if __name__ == "__main__":
    check_source()
    check_host()
    check_original_counterexamples()
    check_storage_alias_regression()
