"""Actual-source CPU interpreter checks with labeled CUDA metadata/context."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / "tritonbench_g/context_attn_llama/context_attn_llama.py"


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
    RemoveIndexWidening().visit(source_kernel(HERE / "context_attn_llama_repaired.py")), include_attributes=False)
spec = importlib.util.spec_from_file_location("context_fwd_repair", HERE / "context_attn_llama_repaired.py")
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


def case(width=16, lengths=(3, 2), prompts=(2, 1), tile=64, read_alias=False):
    heads, kv_heads, batch = 4, 2, len(lengths)
    starts, cursor = [], 1
    for length in lengths:
        starts.append(cursor)
        cursor += length + 2
    total, cache_rows = cursor + 1, 29
    q = torch.randn(total, heads, width) * .1
    k = torch.randn(cache_rows, kv_heads, width) * .1
    v = k if read_alias else torch.randn_like(k) * .1
    backing = torch.full((total + 2, heads, width), 37.)
    out = backing[1:-1]
    seq = [length+prompt for length, prompt in zip(lengths, prompts)]
    tokens = max(seq, default=0)+5
    requests = list(range(batch))[::-1]
    table = torch.randint(cache_rows, (max(batch, 1), tokens), dtype=torch.int32)
    index = [torch.tensor(x, dtype=torch.int32) for x in (requests, starts, seq, prompts)]
    args = [Tensor(x) for x in (q, k, v, out, *index)] + [max(lengths, default=0)+100, Tensor(table)]
    before = [x.clone() for x in (q, k, v, *index, table)]
    Device.name = "Tesla interpreter facade" if tile == 64 else "CPU interpreter facade"
    count = Launch.count
    assert repair.context_attention_fwd(*args) is None
    assert Launch.count-count == bool(any(lengths))
    assert Device.current == torch.device("cuda:1")
    touched = torch.zeros(total, dtype=torch.bool)
    scale = torch.tensor(1.0 / (width ** .5) * 1.4426950408889634, dtype=torch.float32)
    for request, start, length, prompt in zip(requests, starts, lengths, prompts):
        for row in range(length):
            touched[start+row] = True
            end = min((row//tile+1)*tile+prompt, length+prompt)
            count = (end+tile-1)//tile*tile
            for head in range(heads):
                keys = torch.zeros(count, width)
                values = torch.zeros(count, width)
                keys[:end] = k[table[request, :end].long(), head//(heads//kv_heads)]
                values[:end] = v[table[request, :end].long(), head//(heads//kv_heads)]
                scores = keys @ q[start+row, head] * scale
                scores[torch.arange(count)>row+prompt] = -1.e8
                weights = torch.exp2(scores-scores.max())
                expected = (weights @ values)/weights.sum()
                torch.testing.assert_close(out[start+row, head], expected, rtol=6e-4, atol=1e-5)
    assert torch.all(out[~touched] == 37.) and torch.all(backing[[0, -1]] == 37.)
    assert all(torch.equal(x, y) for x, y in zip((q, k, v, *index, table), before))
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
for tile in (64, 128):
    for width in (16, 32, 64, 128, 256):
        case(width=width, tile=tile)
case(lengths=(129,), prompts=(7,), tile=64)
case(lengths=(0, 0), prompts=(0, 1))
case(lengths=(), prompts=())
case(read_alias=True)
args = case()
reject(args, 0, Tensor(torch._neg_view(args[0].value)))
reject(args, 0, Tensor(args[0].value.to(torch.float16)))
reject(args, 1, Tensor(args[1].value, device=torch.device("cuda:1")))
reject(args, 3, args[0])
reject(args, 4, Tensor(torch.tensor([-1, 0], dtype=torch.int32)))
reject(args, 4, Tensor(torch.tensor([100, 0], dtype=torch.int32)))
reject(args, 5, Tensor(torch.tensor([-1, 6], dtype=torch.int32)))
reject(args, 5, Tensor(torch.tensor([1, 2], dtype=torch.int32)))
reject(args, 6, Tensor(torch.tensor([-1, 3], dtype=torch.int32)))
reject(args, 6, Tensor(torch.tensor([1000, 3], dtype=torch.int32)))
reject(args, 7, Tensor(torch.tensor([6, 1], dtype=torch.int32)))
reject(args, 8, 1)
reject(args, 8, True)
bad_table = args[9].value.clone(); bad_table[1, 0] = -1
reject(args, 9, Tensor(bad_table))
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
overlap = list(args); overlap[0], overlap[3] = Tensor(q_alias), Tensor(out_alias)
reject(overlap, 3, overlap[3])
# Finite sentinel semantics deliberately retain the original noncausal tail contribution.
Device.name = "Tesla interpreter facade"
q = torch.zeros(3, 1, 16); q[:, :, 0] = -1.e8
k = torch.zeros_like(q); k[:, :, 0] = 8.
v = torch.zeros_like(q); v[1] = 2.; v[2] = 3.
out = torch.empty_like(q)
ints = [torch.tensor(x, dtype=torch.int32) for x in ([0], [0], [3], [0])]
repair.context_attention_fwd(*[Tensor(x) for x in (q, k, v, out, *ints)], 3,
                             Tensor(torch.tensor([[0, 1, 2]], dtype=torch.int32)))
torch.testing.assert_close(out[0], torch.full_like(out[0], 5/63), rtol=1e-6, atol=1e-6)
print("Llama: original floating AST retained; actual CPU Triton replay passed all widths/tiles, grouped heads, page gather, prompt cache, multi-tile tail, frames, empty calls, read aliases, finite sentinel, device context, and guards")
