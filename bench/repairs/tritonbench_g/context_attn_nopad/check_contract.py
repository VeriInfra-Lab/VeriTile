"""Actual-source CPU interpreter checks with labeled CUDA metadata/context."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / "tritonbench_g/context_attn_nopad/context_attn_nopad.py"


class RemoveIndexWidening(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "to" and
                len(node.args) == 1 and isinstance(node.args[0], ast.Attribute) and
                node.args[0].attr == "int64"):
            return node.func.value
        return node


def source_kernel(path):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_fwd_kernel")


assert ast.dump(source_kernel(ORIGINAL), include_attributes=False) == ast.dump(
    RemoveIndexWidening().visit(source_kernel(HERE / "context_attn_nopad_repaired.py")), include_attributes=False)
spec = importlib.util.spec_from_file_location("context_fwd_repair", HERE / "context_attn_nopad_repaired.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class Tensor:
    def __init__(self, value, **overrides):
        self.value = value
        self.__dict__.update(overrides)
    def __getattr__(self, name):
        if name == "is_cuda": return True
        if name == "device": return torch.device("cuda:0")
        return getattr(self.value, name)


class Device:
    current = torch.device("cuda:1")
    name = "Tesla interpreter facade"
    def __init__(self, owner): self.owner = owner
    def __enter__(self):
        self.previous, Device.current = Device.current, self.owner
    def __exit__(self, *args): Device.current = self.previous


class Launch:
    count = 0
    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            assert Device.current == torch.device("cuda:0")
            Launch.count += 1
            kernel[grid](*[x.value if isinstance(x, Tensor) else x for x in args], **kwargs)
        return invoke


kernel = repair._fwd_kernel
repair._fwd_kernel = Launch()
repair.torch = SimpleNamespace(Tensor=Tensor, float32=torch.float32, int32=torch.int32,
    strided=torch.strided, cuda=SimpleNamespace(device=Device, get_device_name=lambda _: Device.name))


def case(width=16, lengths=(3, 2), gap=2, read_alias=False):
    heads, batch = 2, len(lengths)
    starts, cursor = [], 1
    for length in lengths:
        starts.append(cursor)
        cursor += length + gap
    total = cursor + 1
    q, k, v = [torch.randn(total + 2, heads, width)[1:-1] * .1 for _ in range(3)]
    if read_alias:
        k = q
    backing = torch.full((total + 2, heads, width), 37.)
    out = backing[1:-1]
    index = [torch.tensor(x, dtype=torch.int32) for x in (starts, lengths)]
    args = [Tensor(x) for x in (q, k, v, out, *index)] + [max(lengths, default=0)+100]
    before = [x.clone() for x in (q, k, v)]
    count = Launch.count
    assert repair.context_attention_fwd(*args) is None
    assert Launch.count-count == bool(any(lengths))
    assert Device.current == torch.device("cuda:1")
    touched = torch.zeros(total, dtype=torch.bool)
    for start, length in zip(starts, lengths):
        for row in range(length):
            touched[start+row] = True
            for head in range(heads):
                scores = k[start:start+row+1, head] @ q[start+row, head] / (width ** .5)
                expected = torch.softmax(scores, 0) @ v[start:start+row+1, head]
                torch.testing.assert_close(out[start+row, head], expected, rtol=4e-4, atol=1e-5)
    assert torch.all(out[~touched] == 37.) and torch.all(backing[[0, -1]] == 37.)
    assert all(torch.equal(x, y) for x, y in zip((q, k, v), before))
    return args


def reject(args, position, replacement):
    changed = list(args)
    changed[position] = replacement
    before = args[3].value.clone()
    count = Launch.count
    try: repair.context_attention_fwd(*changed)
    except (TypeError, ValueError):
        assert Launch.count == count
        assert torch.equal(args[3].value, before)
        assert Device.current == torch.device("cuda:1")
    else: raise AssertionError("unsupported input was admitted")


torch.manual_seed(0)
for width in (16, 32, 64, 128):
    case(width)
case(lengths=(129,))
case(lengths=(0, 0))
case(lengths=())
case(read_alias=True)
args = case()
reject(args, 0, Tensor(torch._neg_view(args[0].value)))
reject(args, 0, Tensor(args[0].value.to(torch.float16)))
reject(args, 1, Tensor(args[1].value, device=torch.device("cuda:1")))
reject(args, 3, args[0])
reject(args, 4, Tensor(torch.tensor([-1, 6], dtype=torch.int32)))
reject(args, 4, Tensor(torch.tensor([1, 2], dtype=torch.int32)))
reject(args, 5, Tensor(torch.tensor([-1, 2], dtype=torch.int32)))
reject(args, 5, Tensor(torch.tensor([1000, 2], dtype=torch.int32)))
reject(args, 6, 1)
reject(args, 6, True)
reject(args, 0, object())
for remainder in (1, 2, 3):
    raw = torch.empty(args[0].numel()*4+4, dtype=torch.uint8)
    unaligned = torch.empty(0, dtype=torch.float32).set_(
        raw.untyped_storage()[remainder:], 0, args[0].shape, args[0].stride())
    assert unaligned.is_contiguous() and unaligned.data_ptr()%4 == remainder
    reject(args, 0, Tensor(unaligned))
raw = torch.zeros(args[0].numel()*4+4, dtype=torch.uint8)
q_alias = torch.empty(0, dtype=torch.float32).set_(raw.untyped_storage(), 0, args[0].shape, args[0].stride())
out_alias = torch.empty(0, dtype=torch.float32).set_(raw.untyped_storage()[4:], 0, args[0].shape, args[0].stride())
overlap = list(args)
overlap[0], overlap[3] = Tensor(q_alias), Tensor(out_alias)
reject(overlap, 3, overlap[3])

# The negative-infinity mask keeps exact causality, even below the finite sentinel.
q = torch.zeros(3, 1, 16); q[:, :, 0] = -1.e8
k = torch.zeros_like(q); k[:, :, 0] = 8.
v = torch.zeros_like(q); v[1] = 2.; v[2] = 3.
out = torch.empty_like(q)
repair.context_attention_fwd(*[Tensor(x) for x in (q, k, v, out,
    torch.tensor([0], dtype=torch.int32), torch.tensor([3], dtype=torch.int32))], 3)
assert torch.equal(out[0], v[0])
print("Nopad: original floating AST retained; actual CPU Triton replay passed all widths, multi-tile tail, packed gaps, frame, empty calls, causal mask, device context, and guards")
