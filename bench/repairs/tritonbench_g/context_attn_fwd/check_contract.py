"""Actual-source CPU interpreter checks with labeled CUDA metadata/context."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / "tritonbench_g/context_attn_fwd/context_attn_fwd.py"


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
                if isinstance(n, ast.FunctionDef) and n.name == "_fwd_kernel_int8kv")


assert ast.dump(source_kernel(ORIGINAL), include_attributes=False) == ast.dump(
    RemoveIndexWidening().visit(source_kernel(HERE / "context_attn_fwd_repaired.py")), include_attributes=False)
spec = importlib.util.spec_from_file_location("context_fwd_repair", HERE / "context_attn_fwd_repaired.py")
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


kernel = repair._fwd_kernel_int8kv
repair._fwd_kernel_int8kv = Launch()
repair.torch = SimpleNamespace(Tensor=Tensor, float32=torch.float32, int32=torch.int32,
    strided=torch.strided, cuda=SimpleNamespace(device=Device, get_device_name=lambda _: Device.name))


def case(block=64, width=16, lengths=(3, 2), prompts=(1, 0)):
    Device.name = "Tesla interpreter facade" if block == 64 else "interpreter facade"
    batch, heads, kv_heads, capacity = len(lengths), 2, 1, max([1] + [a+b for a, b in zip(lengths, prompts)])
    starts = [sum(lengths[:i]) for i in range(batch)]
    total = sum(lengths)
    q = torch.randn(total + 2, heads, width) * .1
    q = q[1:-1]
    # Adjacent K/V head slices retain a batch gap and shared read-only backing.
    cache = torch.randn(batch, 2*kv_heads, capacity, width) * .1
    k, v = cache[:, :kv_heads], cache[:, kv_heads:]
    backing = torch.full((total + 2, heads, width), 37.)
    out = backing[1:-1]
    index = [torch.tensor(values, dtype=torch.int32) for values in
             (starts, [a+b for a, b in zip(lengths, prompts)], prompts)]
    before_q, before_cache = q.clone(), cache.clone()
    args = [Tensor(x) for x in (q, k, v, out, index[0], index[1])]
    args += [max(lengths, default=0), Tensor(index[2])]
    calls = Launch.count
    assert repair.context_attention_fwd_ppl_int8kv(*args) is None
    assert Launch.count-calls == bool(total and batch)
    assert Device.current == torch.device("cuda:1")
    scale = 1.0 / (width ** .5) * 1.4426950408889634
    for b, length in enumerate(lengths):
        for row in range(length):
            for head in range(heads):
                end = min((row//block+1)*block+prompts[b], length+prompts[b])
                padded = ((end+block-1)//block)*block
                scores = torch.full((padded,), -1.e8)
                scores[:row+prompts[b]+1] = (k[b, head//heads, :row+prompts[b]+1] @ q[starts[b]+row, head]) * scale
                weights = torch.exp2(scores-scores.max())
                expected = weights[:end] @ v[b, head//heads, :end] / weights.sum()
                torch.testing.assert_close(out[starts[b]+row, head], expected, rtol=4e-4, atol=1e-5)
    assert torch.equal(q, before_q) and torch.equal(cache, before_cache)
    assert torch.all(backing[[0, -1]] == 37.)
    return args


def reject(args, position, replacement):
    changed = list(args)
    changed[position] = replacement
    before = args[3].value.clone()
    count = Launch.count
    try: repair.context_attention_fwd_ppl_int8kv(*changed)
    except (TypeError, ValueError):
        assert Launch.count == count
        assert torch.equal(args[3].value, before)
        assert Device.current == torch.device("cuda:1")
    else: raise AssertionError("unsupported input was admitted")


torch.manual_seed(0)
for block in (64, 128):
    case(block)
case(64, 32, (65,), (3,))
case(128, 128, (1,), (1,))
case(64, 256, (1,), (0,))
case(lengths=(0, 0), prompts=(1, 0))
case(lengths=(), prompts=())
args = case()
reject(args, 0, Tensor(torch._neg_view(args[0].value)))
reject(args, 0, Tensor(args[0].value.to(torch.float16)))
reject(args, 1, Tensor(args[1].value, device=torch.device("cuda:1")))
reject(args, 3, args[0])
reject(args, 4, Tensor(torch.tensor([-1, 3], dtype=torch.int32)))
reject(args, 4, Tensor(torch.tensor([0, 1], dtype=torch.int32)))
reject(args, 5, Tensor(torch.tensor([0, 2], dtype=torch.int32)))
reject(args, 6, 1)
reject(args, 6, True)
reject(args, 0, object())
for remainder in (1, 2, 3):
    raw = torch.empty(args[0].numel()*4+4, dtype=torch.uint8)
    unaligned = torch.empty(0, dtype=torch.float32).set_(
        raw.untyped_storage()[remainder:], 0, args[0].shape, args[0].stride())
    assert unaligned.is_contiguous() and unaligned.data_ptr()%4 == remainder
    reject(args, 0, Tensor(unaligned))

# Distinct external storage objects can still cover overlapping byte ranges.
raw = torch.zeros(args[0].numel()*4+4, dtype=torch.uint8)
q_alias = torch.empty(0, dtype=torch.float32).set_(raw.untyped_storage(), 0, args[0].shape, args[0].stride())
out_alias = torch.empty(0, dtype=torch.float32).set_(raw.untyped_storage()[4:], 0, args[0].shape, args[0].stride())
assert q_alias.untyped_storage().data_ptr() != out_alias.untyped_storage().data_ptr()
overlap_args = list(args)
overlap_args[0], overlap_args[3] = Tensor(q_alias), Tensor(out_alias)
reject(overlap_args, 3, overlap_args[3])

# Finite masking is observable when valid causal scores are below -1e8.
Device.name = "Tesla interpreter facade"
q = torch.zeros(3, 1, 16); q[:, :, 0] = -1.e8
k = torch.zeros(1, 1, 3, 16); k[:, :, :, 0] = 8.
v = torch.zeros_like(k); v[0, 0, 1] = 2.; v[0, 0, 2] = 3.
out = torch.empty_like(q)
repair.context_attention_fwd_ppl_int8kv(*[Tensor(x) for x in
    (q, k, v, out, torch.tensor([0], dtype=torch.int32), torch.tensor([3], dtype=torch.int32))],
    3, Tensor(torch.tensor([0], dtype=torch.int32)))
torch.testing.assert_close(out[0], torch.full((1, 16), 5./63), rtol=1e-5, atol=1e-6)
print("Context attention fwd: exact floating-source AST retained; actual Triton replay passed both tiles, widths/tails, grouped cached attention, finite sentinel, frame, empty calls and guard rejection")
