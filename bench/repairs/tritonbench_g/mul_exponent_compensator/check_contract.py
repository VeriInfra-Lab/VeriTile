"""Source/guard and physical CPU-storage regression checks; no GPU numerics claim."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "mul_exponent_compensator_repaired.py"
ROOT = HERE.parents[3]
ORIGINAL = ROOT / "bench/tritonbench_g/mul_exponent_compensator/mul_exponent_compensator.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helpers = load_file("add_example_helpers", HERE.parent / "add_example/check_contract.py")
TensorFacade, Pointer, raw_storage = helpers.TensorFacade, helpers.Pointer, helpers.raw_storage


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_binding():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == "d1978613929a72a7d01dc152e095143a27abfe01be68071c845e38468b086027"
    kernel = function(SOURCE, "mul_kernel_repaired")
    actual = ast.unparse(ast.Module(body=kernel.body, type_ignores=[]))
    expected = '''exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
idxs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
mask = idxs < n_elements
x = tl.load(src + idxs, mask=mask)
y = x * exponent_compensator
tl.store(dst + idxs, y, mask=mask)'''
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    wrapper = function(SOURCE, "launch_mul_kernel")
    assert ast.unparse(wrapper.args.defaults[0]) == "1"
    guards = [ast.unparse(n.test) for n in wrapper.body if isinstance(n, ast.If)]
    assert guards == [
        "not isinstance(src, torch.Tensor)", "not isinstance(BLOCK_SIZE, int)",
        "BLOCK_SIZE not in tuple((1 << k for k in range(21)))", "src.layout != torch.strided",
        "src.dtype != torch.float32", "not src.is_cuda", "not src.is_contiguous()", "src.is_neg()",
        "triton.cdiv(n_elements, BLOCK_SIZE) > 2 ** 31 - 1", "n_elements == 0",
        "src.device != dst.device", "src.data_ptr() % 4 != 0"]
    assert ast.unparse(wrapper.body[1]) == "dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')"
    assert ast.unparse(wrapper.body[2]) == "src.shape[0]"
    lean = (HERE / "Contract.lean").read_text()
    for guard in ['x.shape = []', 'B ∉ supportedBlocks', 'x.layout != .strided',
                  'x.dtype != .floating .fp32', '!x.isCuda', '!x.isContiguous', 'x.isNeg',
                  '(x.numel+B-1)/B > 2^31-1', 'x.numel > 0 && x.device != ⟨"cuda",current⟩',
                  'x.numel > 0 && x.dataPtr.toNat%4 != 0']:
        assert guard in lean
    assert 'List.range 21' in lean and 'map (fun k => 2^k)' in lean
    assert not any(n.attr in {"contiguous", "flatten", "clone", "reshape", "resolve_neg"}
                   for n in ast.walk(wrapper) if isinstance(n, ast.Attribute))
    assert '.Proofs.' not in (HERE / "Kernels.lean").read_text()


def interpreter():
    current = [0]
    def load(p, mask):
        result = torch.zeros_like(p.offsets, dtype=torch.float32)
        result[mask] = raw_storage(p.tensor)[p.tensor.storage_offset() + p.offsets[mask]]
        return result
    def store(p, value, mask):
        raw_storage(p.tensor)[p.tensor.storage_offset() + p.offsets[mask]] = value[mask]
    kernel = function(SOURCE, "mul_kernel_repaired")
    kernel.decorator_list = []
    for arg in kernel.args.args:
        arg.annotation = None
    code = ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[]))
    ns = {"tl": SimpleNamespace(program_id=lambda axis: torch.tensor(current[0], dtype=torch.int32),
                                 arange=lambda lo, hi: torch.arange(lo, hi, dtype=torch.int32),
                                 int64=torch.int64, constexpr=float, load=load, store=store)}
    exec(compile(code, str(SOURCE), "exec"), ns)
    return ns["mul_kernel_repaired"], current


def checks():
    module = load_file("mul_exponent_contract", SOURCE)
    kernel, pid = interpreter()
    calls, allocated = [], []
    current_device = [0]
    def empty(shape, dtype, device):
        assert dtype == torch.float32 and device == 'cuda'
        calls.append("allocation")
        count = torch.Size(shape).numel()
        if count > 100000:
            result = TensorFacade(torch.empty(shape, device="meta"), device=f"cuda:{current_device[0]}")
        else:
            backing = torch.full((count+8,), -123., dtype=dtype)
            result = TensorFacade(backing[3:3+count].view(shape), device=f"cuda:{current_device[0]}")
        allocated.append(result)
        return result
    class Launch:
        def __getitem__(self, grid):
            def run(src, dst, n, block):
                calls.append((grid, src, dst, n, block))
                for p in reversed(range(grid[0])):
                    pid[0] = p
                    kernel(Pointer(src.data), Pointer(dst.data), n, block)
            return run
    module.torch = SimpleNamespace(Tensor=TensorFacade, float32=torch.float32, strided=torch.strided,
        empty=empty, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.mul_kernel_repaired = Launch()
    valid = TensorFacade(torch.ones(3))
    bad = [([], 1, TypeError, "Tensor", False),
        (TensorFacade(torch.tensor(3.)), 0, IndexError, "", True),
        (valid, 1., TypeError, "integer", True), (valid, "1", TypeError, "integer", True),
        (valid, 0, ValueError, "power of two", True), (valid, -1, ValueError, "power of two", True),
        (valid, 3, ValueError, "power of two", True), (valid, 1<<21, ValueError, "power of two", True),
        (valid, False, ValueError, "power of two", True),
        (TensorFacade(torch.eye(2).to_sparse()), 1, TypeError, "strided", True),
        (TensorFacade(torch.ones(3, dtype=torch.float64)), 1, TypeError, "float32", True),
        (TensorFacade(torch.ones(3), device="cpu", cuda=False), 1, ValueError, "CUDA", True),
        (TensorFacade(torch.ones(6)[::2]), 1, ValueError, "contiguous", True),
        (TensorFacade(torch._neg_view(torch.ones(3))), 1, ValueError, "negative view", True),
        (TensorFacade(torch.empty(1<<31, device="meta")), 1, ValueError, "grid", True),
        (TensorFacade(torch.ones(3), device="cuda:1"), 1, ValueError, "current output device", True)]
    for offset in (1, 2, 3):
        bad.append((TensorFacade(helpers.unaligned_tensor(offset)), 4, ValueError, "4-byte", True))
    for src, block, err, message, after_allocation in bad:
        calls.clear()
        try: module.launch_mul_kernel(src, block)
        except err as caught: assert message in str(caught)
        else: raise AssertionError("invalid argument accepted")
        assert calls == (["allocation"] if after_allocation else [])
    backing = torch.arange(20, dtype=torch.float32)
    good = [(torch.arange(3, dtype=torch.float32), 8),
        (torch.arange(5, dtype=torch.float32), 4),
        (torch.arange(6, dtype=torch.float32).view(2, 3), 4),
        (backing[3:8], 4), (torch.as_strided(backing, (1, 4), (99, 1), 2), 2),
        (torch.empty(2, 0), 4), (torch.as_strided(backing, (0,), (1,), 1000), 4),
        (helpers.unaligned_tensor(1, empty=True), 8),
        (torch.tensor([-0., 0., 1e-38, float("inf"), -float("inf"), float("nan")]), 8),
        (torch.ones(3), True), (torch.ones(1), 1<<20)]
    for data, block in good:
        calls.clear()
        src = TensorFacade(data)
        saved = raw_storage(data).view(torch.uint8).clone()
        out = module.launch_mul_kernel(src, block)
        assert out is allocated[-1] and out is not src
        assert out.shape == src.shape and out.device == torch.device("cuda:0")
        torch.testing.assert_close(out.data, data*2.**112, atol=0, rtol=0, equal_nan=True)
        assert torch.equal(raw_storage(data).view(torch.uint8), saved)
        storage = raw_storage(out.data)
        assert torch.equal(storage[:3], torch.full((3,), -123.))
        assert torch.equal(storage[3+data.numel():], torch.full((5,), -123.))
        if data.numel():
            assert len(calls) == 2
            grid, actual_src, actual_dst, n, actual_block = calls[-1]
            assert actual_src is src and actual_dst is out
            assert (grid, n, actual_block) == (((data.numel()+block-1)//block,), data.numel(), block)
        else: assert calls == ["allocation"]
    # Original device='cuda' means current output device, including empty calls.
    current_device[0] = 1
    calls.clear()
    out = module.launch_mul_kernel(TensorFacade(torch.empty(0), device="cuda:0"))
    assert out.device == torch.device("cuda:1") and calls == ["allocation"]
    calls.clear()
    src = TensorFacade(torch.ones(3), device="cuda:1")
    out = module.launch_mul_kernel(src)
    assert out.device == torch.device("cuda:1") and calls[-1][-1] == 1
    # Original floor-grid misses all short inputs, a nonmultiple tail, and rank>1 coordinates.
    for shape, block in [((3,), 4), ((5,), 4), ((2, 3), 1)]:
        covered = (shape[0]//block)*block
        assert covered < torch.Size(shape).numel()
    # The widened product retains indices beyond signed32 instead of wrapping.
    index = torch.tensor(1<<30, dtype=torch.int32).to(torch.int64)*4
    assert index.item() == 1<<32
    assert torch.tensor(1<<30, dtype=torch.int32).mul(4).item() == 0
    print(f"Source/guard checks; {len(bad)} rejection and {len(good)+2} admission cases passed.")
    print("Actual JIT AST interpreted over CPU storage, reverse program order, input/exterior byte frames checked.")
    print(f"CUDA available: {torch.cuda.is_available()}; numerical GPU validation remains pending.")

if __name__ == "__main__":
    source_binding()
    checks()
