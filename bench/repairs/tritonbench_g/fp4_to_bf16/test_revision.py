"""CPU source interpretation and checked wrapper admission; never launches CUDA."""
import ast
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fp4_to_bf16_repaired.py"
spec = importlib.util.spec_from_file_location("fp4_checked", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference(packed):
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.bfloat16)
    return torch.stack((table[(packed >> 4).long()], table[(packed & 15).long()]), 1).flatten()


for n, offset in [(1, 0), (15, 3), (256, 17), (511, 7), (512, 0), (513, 17), (1025, 1)]:
    storage = (torch.arange(n + offset + 9) % 256).to(torch.uint8)
    before = storage.clone()
    x = storage[offset:offset+n]
    guard = torch.full((2*n+14,), 0x1234, dtype=torch.int16)
    y = guard[7:7+2*n].view(torch.bfloat16)
    module.triton_f4_to_bf16_kernel_repaired[(triton.cdiv(n, 512),)](
        x, y, n, sign_mask_f4=8, mantissa_mask_f4=1, mbits_f4_e2m1=1,
        ebits_f4_e2m1=2, f4_e2m1_exp_bias=1, mbits_f32=23, ebits_f32=8,
        f32_exp_bias=127, zero_bits_f32=0, zero_point_five_bits_f32=0x3F000000,
        BLOCK_SIZE_IN=512)
    assert torch.equal(y.view(torch.int16), reference(x).view(torch.int16))
    assert torch.equal(storage, before)
    assert torch.all(guard[:7] == 0x1234) and torch.all(guard[-7:] == 0x1234)

tree = ast.parse(SOURCE.read_text())
host = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == "triton_f4_to_bf16_repaired")
constants = [n for n in tree.body if isinstance(n, ast.Assign)]
code = compile(ast.Module(body=constants+[host], type_ignores=[]), str(SOURCE), "exec")
calls = []
current_device, launch_devices, fail_launch = ["cuda:0"], [], [False]


class DeviceContext:
    def __init__(self, device): self.device = device
    def __enter__(self):
        self.previous = current_device[0]
        current_device[0] = self.device
    def __exit__(self, *exc): current_device[0] = self.previous


class Tensor:
    def __init__(self, shape=(513,), *, dtype="uint8", layout="strided", cuda=True,
                 contiguous=True, negative=False):
        self.shape, self.dtype, self.layout = shape, dtype, layout
        self.is_cuda, self.c, self.negative = cuda, contiguous, negative
        self.device, self.ndim = "cuda:1" if cuda else "cpu", len(shape)
    def is_contiguous(self): return self.c
    def is_neg(self): return self.negative
    def numel(self): return math.prod(self.shape)


def empty(*shape, device, dtype):
    output = Tensor(shape, dtype=dtype)
    output.device = device
    calls.append(("allocate", output))
    return output


class Kernel:
    def __getitem__(self, grid):
        def launch(x, output, count, **kwargs):
            assert current_device[0] == x.device == output.device
            launch_devices.append(current_device[0])
            if fail_launch[0]: raise RuntimeError("mock submission failure")
            assert output is calls[0][1]
            assert count == math.prod(x.shape)
            assert grid(kwargs) == ((count + 511)//512,)
            calls.append(("launch", count))
        return launch


env = dict(torch=SimpleNamespace(Tensor=Tensor, strided="strided", uint8="uint8",
           bfloat16="bf16", empty=empty, cuda=SimpleNamespace(device=DeviceContext)), triton=triton,
           triton_f4_to_bf16_kernel_repaired=Kernel())
exec(code, env)
checked = env[host.name]
bad = [object(), Tensor(layout="sparse"), Tensor(dtype="int8"), Tensor(dtype="fp32"),
       Tensor(shape=()), Tensor(cuda=False), Tensor(contiguous=False), Tensor(negative=True)]
for x in bad:
    calls.clear()
    try:
        checked(x)
        raise AssertionError("accepted unsupported metadata")
    except (TypeError, ValueError):
        assert calls == []

good = [Tensor((0,)), Tensor((0,513)), Tensor((1,)), Tensor((2,3)), Tensor((513,)),
        Tensor((2**30,)), Tensor((2**31,)), Tensor(((2**63-1)//4,))]
for x in good:
    calls.clear()
    output = checked(x)
    assert output is calls[0][1]
    assert output.shape == (*x.shape[:-1], 2*x.shape[-1])
    assert output.dtype == "bf16" and output.device == x.device
    assert len(calls) == (1 if x.numel() == 0 else 2)
    assert current_device[0] == "cuda:0"
    if x.numel(): assert launch_devices[-1] == "cuda:1"

calls.clear()
fail_launch[0] = True
try:
    checked(Tensor())
except RuntimeError:
    pass
else:
    raise AssertionError("expected mock submission failure")
finally:
    fail_launch[0] = False
assert current_device[0] == "cuda:0" and launch_devices[-1] == "cuda:1"

# Count boundaries above are metadata-only controls; ordinary allocation/grid
# rejection remains. A uint8 lazy-negative view has different logical bytes.
raw = torch.tensor([1], dtype=torch.uint8)
logical = torch._neg_view(raw)
assert logical.is_neg() and logical.item() == 255 and raw.item() == 1
for n, succeeds in [((2**63-1)//4, True), (2**61, False)]:
    try:
        out = torch.empty((2*n,), dtype=torch.bfloat16, device="meta")
        assert succeeds and out.untyped_storage().nbytes() == 4*n
    except RuntimeError:
        assert not succeeds

print({"raw_word_and_frame_cases": 7, "guard_rejections": len(bad),
       "wrapper_admissions": len(good), "storage_capacity_boundaries": 2,
       "owning_device_exception_restoration": True,
       "gpu_execution": False})
