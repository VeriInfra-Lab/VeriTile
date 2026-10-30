"""Actual CPU JIT interpretation plus wrapper/API controls; no GPU launch."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("f8_checked", HERE / "f8_conversion_utils_repaired.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
limitations, source_cases = {}, 0
for fmt, dtype in [(module.tl.float8e5, torch.float8_e5m2),
                   (module.tl.float8e4nv, torch.float8_e4m3fn)]:
    storage = torch.arange(290).to(torch.uint8).view(torch.int8)
    x = storage[17:273]
    saved = storage.clone()
    output = torch.full((270,), 0x1234, dtype=torch.int16)
    y = output[7:263].view(torch.float16)
    module.kernel_f8_to_f16[(1,)](y, triton.reinterpret(x, fmt), 256, BLOCK_SIZE=1024)
    reference = x.view(dtype).to(torch.float16)
    bad = (y != reference) & ~(y.isnan() & reference.isnan())
    limitations[str(fmt)] = x.view(torch.uint8)[bad].tolist()
    # Installed Triton CPU interpretation is not native FP8 conformance:
    # E5 subnormals and E4 NaN codes differ. Keep these visible, not excluded
    # from the checked input contract or hidden as structural obligations.
    permitted = [1, 2, 3, 129, 130, 131] if fmt is module.tl.float8e5 else [127, 255]
    assert set(limitations[str(fmt)]).issubset(permitted)
    good = ~bad & ~reference.isnan()
    assert torch.equal(y[good].view(torch.int16), reference[good].view(torch.int16))
    assert torch.equal(storage, saved)
    assert torch.all(output[:7] == 0x1234) and torch.all(output[-7:] == 0x1234)
    source_cases += 1
    for precision in [torch.float16, torch.float32]:
        for n in [1, 15, 1023, 1024, 1025, 2051]:
            values = torch.tensor([-8., -1.5, -1., -0., 0., .5, 1., 1.5, 8.], dtype=precision)
            storage = values.repeat((n+39)//9+1)
            x = storage[17:17+n]
            saved = storage.clone()
            output = torch.full((n+14,), 37, dtype=torch.int8)
            y = output[7:7+n]
            module.kernel_f16_to_f8[(triton.cdiv(n,1024),)](
                triton.reinterpret(y,fmt), x, n, BLOCK_SIZE=1024)
            assert torch.equal(y, x.to(dtype).view(torch.int8))
            assert torch.equal(storage, saved)
            assert torch.all(output[:7] == 37) and torch.all(output[-7:] == 37)
            source_cases += 1

actual_torch = torch
widths = {"int8": 1, "fp16": 2, "fp32": 4}
calls, current = [], [0]


class Tensor:
    def __init__(self, shape=(1025,), dtype="fp32", *, layout="strided", cuda=True,
                 contiguous=True, negative=False, pointer=4096, origin=None, storage_numel=None):
        self.shape, self.dtype, self.layout = shape, dtype, layout
        self.is_cuda, self.contiguous, self.negative = cuda, contiguous, negative
        self.device, self.pointer = "cuda:2" if cuda else "cpu", pointer
        self.origin = self if origin is None else origin
        self.storage_numel = self.numel() if storage_numel is None else storage_numel
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self.contiguous
    def is_neg(self): return self.negative
    def data_ptr(self): return self.pointer
    def element_size(self): return widths[self.dtype]
    def untyped_storage(self): return SimpleNamespace(size=lambda: self.storage_numel*self.element_size())
    def view(self, shape):
        assert shape == -1 and self.contiguous
        return Tensor((self.numel(),), self.dtype, pointer=self.pointer, origin=self.origin)
    def __getitem__(self, index):
        assert isinstance(index, slice) and index.step is None
        assert 0 <= index.start <= index.stop <= self.numel()
        return Tensor((index.stop-index.start,), self.dtype,
                      pointer=self.pointer+index.start*self.element_size(), origin=self.origin)


class Device:
    def __init__(self, device): self.device = device
    def __enter__(self):
        assert self.device == "cuda:2"
        self.previous = current[0]
        current[0] = 2
    def __exit__(self, *args): current[0] = self.previous


def empty_like(x, dtype):
    result = Tensor(x.shape, dtype, pointer=1 << 50)
    calls.append(("allocation",result))
    return result


def reinterpret(x, dtype):
    return SimpleNamespace(tensor=x, dtype=dtype)


class Kernel:
    def __getitem__(self, grid):
        def launch(y, x, count, **kwargs):
            xx, yy = getattr(x,"tensor",x), getattr(y,"tensor",y)
            calls.append(("launch",count,xx,yy))
            if callable(grid): grid_value=grid(kwargs)
            else: grid_value=grid
            assert grid_value == ((xx.numel()+1023)//1024,)
            assert kwargs == {"BLOCK_SIZE":1024}
        return launch


fake_torch = SimpleNamespace(Tensor=Tensor,strided="strided",int8="int8",float16="fp16",
    float32="fp32",empty_like=empty_like,cuda=SimpleNamespace(device=Device))
module.torch = fake_torch
module.triton = SimpleNamespace(cdiv=triton.cdiv,reinterpret=reinterpret)
module.kernel_f8_to_f16 = Kernel()
module.kernel_f16_to_f8 = Kernel()
rejections, admissions = 0, 0
for decode in [False,True]:
    dtype = "int8" if decode else "fp32"
    wrapper = module.f8_to_f16_repaired if decode else module.f16_to_f8_repaired
    bad = [object(), Tensor(dtype=dtype,layout="sparse"), Tensor(dtype="fp32" if decode else "int8"),
           Tensor(dtype=dtype,cuda=False),Tensor(dtype=dtype,contiguous=False),
           Tensor(dtype=dtype,negative=True)]
    if not decode: bad += [Tensor(dtype=dtype,pointer=4097)]
    for x in bad:
        calls.clear()
        try:
            wrapper(x)
            raise AssertionError("unsupported input admitted")
        except (TypeError,ValueError):
            assert not calls and current[0] == 0
        rejections += 1
    calls.clear()
    try:
        wrapper(Tensor(dtype=dtype), module.tl.float32)
        raise AssertionError("unsupported format admitted")
    except TypeError:
        assert not calls
    rejections += 1
    for dt in (["int8"] if decode else ["fp16","fp32"]):
        for shape in [(),(0,),(0,17),(1,),(2,3),(1025,),(2**30+1,),(2**31+1,)]:
            for fmt in [module.tl.float8e5,module.tl.float8e4nv]:
                calls.clear()
                x=Tensor(shape,dt,pointer=4096+17*widths[dt])
                result=wrapper(x,fmt)
                assert result is calls[0][1] and result.shape == shape
                assert result.dtype == ("fp16" if decode else "int8")
                assert result.device == x.device and current[0] == 0
                start=0
                for _,count,xx,yy in calls[1:]:
                    assert count == min(module.CHUNK_SIZE,x.numel()-start)
                    assert xx.origin is x and yy.origin is result
                    assert xx.data_ptr() == x.data_ptr()+start*x.element_size()
                    assert yy.data_ptr() == result.data_ptr()+start*result.element_size()
                    start += count
                assert start == x.numel()
                admissions += 1

# Execute the original wrapper AST to retain the concrete storage-count bug.
original=HERE.parents[2]/"tritonbench_g/f8_conversion_utils/f8_conversion_utils.py"
tree=ast.parse(original.read_text())
host=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="f16_to_f8")
env=dict(torch=fake_torch,triton=module.triton,tl=module.tl,kernel_f16_to_f8=Kernel())
exec(compile(ast.Module(body=[host],type_ignores=[]),str(original),"exec"),env)
calls.clear()
x=Tensor((3,),"fp32",storage_numel=8)
env[host.name](x)
assert calls[1][1] == 8 and calls[1][3].numel() == 3
calls.clear()
module.f16_to_f8_repaired(x)
assert calls[1][1] == 3

report=dict(source_interpreter_cases=source_cases,guard_rejections=rejections,
    wrapper_admissions=admissions,original_counterexamples=1,
    interpreter_scalar_mismatches=limitations,gpu_execution=False)
(HERE/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(report)
