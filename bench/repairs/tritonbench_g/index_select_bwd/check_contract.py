"""Actual repaired Triton source replay; CUDA metadata/context are mocked."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bwd_repair", HERE / "index_select_bwd_repaired.py")
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

    def as_strided(self, *args, **kwargs):
        return Tensor(self.value.as_strided(*args, **kwargs))

    def __getitem__(self, key):
        return Tensor(self.value[key])


class Launch:
    count = 0
    def __getitem__(self, grid):
        assert grid == (1,)
        def launch(*args, **kwargs):
            assert DeviceContext.current == torch.device("cuda:0")
            Launch.count += 1
            kernel[grid](*[x.value if isinstance(x, Tensor) else x for x in args], **kwargs)
        return launch


class DeviceContext:
    current = torch.device("cuda:1")
    def __init__(self, device):
        self.device = device
    def __enter__(self):
        self.previous = DeviceContext.current
        DeviceContext.current = self.device
    def __exit__(self, *args):
        DeviceContext.current = self.previous


kernel = repair.index_select_cat_bwd_repaired_kernel
repair.index_select_cat_bwd_repaired_kernel = Launch()
repair.torch = SimpleNamespace(Tensor=Tensor, float32=torch.float32, int64=torch.int64,
    strided=torch.strided, cuda=SimpleNamespace(device=DeviceContext))


case_count = 0
empty_count = 0
rejection_count = 0

def case(ids, cols, rows=7, singleton=False):
    global case_count, empty_count
    case_count += 1
    empty_count += not ids or cols == 0
    backing = torch.arange((rows + 2) * cols, dtype=torch.float32).reshape(rows + 2, cols)
    destination = backing[1:-1]
    incoming_backing = (torch.arange((len(ids) + 2) * cols, dtype=torch.float32) / 8).reshape(len(ids) + 2, cols)
    incoming = incoming_backing[1:-1]
    if singleton and cols == 1:
        destination = destination.as_strided(destination.shape, (1, 17))
        incoming = incoming.as_strided(incoming.shape, (1, 17))
    index = torch.tensor([99] + ids + [99], dtype=torch.int64)[1:-1]
    before = backing.clone()
    incoming_before = incoming_backing.clone()
    expected = destination.clone()
    if cols:
        for i, row in enumerate(ids):
            expected[row] += incoming[i]
    args = (Tensor(destination), Tensor(index), Tensor(incoming))
    launches = Launch.count
    assert repair.index_select_cat_bwd_repaired(*args) is None
    assert DeviceContext.current == torch.device("cuda:1")
    assert Launch.count - launches == len(ids) * ((cols + 511) // 512)
    assert torch.equal(destination.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(backing[[0, -1]], before[[0, -1]])
    assert torch.equal(incoming_backing, incoming_before)
    assert torch.equal(index, torch.tensor(ids, dtype=torch.int64))
    return args


def reject(args, position, value):
    global rejection_count
    rejection_count += 1
    changed = list(args)
    changed[position] = value
    before = args[0].value.clone()
    launches = Launch.count
    try: repair.index_select_cat_bwd_repaired(*changed)
    except (TypeError, ValueError, IndexError):
        assert Launch.count == launches
        assert torch.equal(args[0].value, before)
    else: raise AssertionError("unsupported input admitted")


for cols in (0, 1, 3, 511, 512, 513, 1025):
    case([], cols)
    case([2, 0, 2, 1, 0], cols, singleton=True)
case([123, -456], 0)
case([], 0, rows=0)

# The first source row retains its nonzero initial value, and the additions
# distinguish the required ascending order from reassociation or overwrite.
destination = torch.tensor([[1.], [23.], [29.]])
incoming = torch.tensor([[2**24], [-2**24], [1.]], dtype=torch.float32)
assert repair.index_select_cat_bwd_repaired(Tensor(destination), Tensor(torch.tensor([0, 0, 0])), Tensor(incoming)) is None
assert destination[0, 0].item() == 1.
assert torch.equal(destination[1:], torch.tensor([[23.], [29.]]))

# Unselected words, including signed zero and NaN payloads, retain every bit.
special = torch.tensor([-2147483648, 2143289345, 1, 2139095040], dtype=torch.int32)
destination = special.view(torch.float32).reshape(4, 1).clone()
repair.index_select_cat_bwd_repaired(Tensor(destination), Tensor(torch.tensor([2])), Tensor(torch.ones(1, 1)))
assert torch.equal(destination[[0, 1, 3]].view(torch.int32).flatten(), special[[0, 1, 3]])

args = case([0, 2], 3)
reject(args, 1, Tensor(torch.tensor([-1, 2])))
reject(args, 1, Tensor(torch.tensor([7, 2])))
reject(args, 1, Tensor(torch.tensor([0, 2], dtype=torch.int32)))
reject(args, 2, Tensor(torch.empty(2, 3, dtype=torch.float16)))
reject(args, 2, Tensor(args[0].value[:2]))
reject(args, 0, Tensor(torch._neg_view(args[0].value)))
reject(args, 2, Tensor(args[2].value, device=torch.device("cuda:1")))
reject(args, 0, Tensor(args[0].value, is_cuda=False))
reject(args, 2, Tensor(torch.empty(8, 3)))
reject(args, 2, Tensor(torch.empty(2, 4)))
reject(args, 1, object())
for remainder in (1, 2, 3):
    bytes_ = torch.empty(args[0].numel() * 4 + 4, dtype=torch.uint8)
    unaligned = torch.empty(0, dtype=torch.float32).set_(
        bytes_.untyped_storage()[remainder:], 0, args[0].shape, args[0].stride())
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 4 == remainder
    reject(args, 0, Tensor(unaligned))
bytes_ = torch.empty(24, dtype=torch.uint8)
unaligned_index = torch.empty(0, dtype=torch.int64).set_(bytes_.untyped_storage()[1:], 0, (2,), (1,))
assert unaligned_index.data_ptr() % 8 == 1
reject(args, 1, Tensor(unaligned_index))
single = case([0, 2], 1)
reject(single, 2, Tensor(single[2].value.as_strided((2, 1), (1, 19))))
# Both read roles may share backing storage. The index words are also the
# incoming FP32 payload bytes; the destination is a separate allocation.
shared = torch.zeros(16, dtype=torch.uint8)
shared_index = shared.view(torch.int64)
shared_index[:] = torch.tensor([0, 2])
shared_incoming = shared.view(torch.float32).reshape(2, 2)
shared_before = shared.clone()
shared_destination = torch.ones(3, 2)
shared_expected = shared_destination.clone()
for j, row in enumerate(shared_index.tolist()):
    shared_expected[row] += shared_incoming[j]
repair.index_select_cat_bwd_repaired(Tensor(shared_destination), Tensor(shared_index), Tensor(shared_incoming))
assert torch.equal(shared_destination.view(torch.int32), shared_expected.view(torch.int32))
assert torch.equal(shared, shared_before)

# Run the unchanged benchmark kernel plus the prior column-product candidate.
# This is the historical overwrite pair, not a candidate for the repair.
original_path = HERE.parents[2] / "tritonbench_g/index_select_bwd/index_select_bwd.py"
module_ast = ast.parse(original_path.read_text())
original_node = next(n for n in module_ast.body if isinstance(n, ast.FunctionDef)
                     and n.name == "index_select_cat_bwd_kernel")
candidate_node = copy.deepcopy(original_node)
candidate_node.name = "index_select_cat_bwd_column_commuted_kernel"
changes = 0
for node in ast.walk(candidate_node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
        if (node.left.id, node.right.id) == ("pid1", "BLOCK_SIZE_COL"):
            node.left, node.right = node.right, node.left
            changes += 1
assert changes == 1
source = "import torch\nimport triton\nimport triton.language as tl\n\n" + ast.unparse(original_node) + "\n\n" + ast.unparse(candidate_node) + "\n"
with tempfile.TemporaryDirectory(prefix="index-select-bwd-original-") as directory:
    file = Path(directory) / "original_pair.py"
    file.write_text(source)
    original_spec = importlib.util.spec_from_file_location("index_bwd_original_pair", file)
    original_pair = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original_pair)
    original_launches = 0
    for selected in (original_pair.index_select_cat_bwd_kernel,
                     original_pair.index_select_cat_bwd_column_commuted_kernel):
        destination = torch.tensor([[5.]])
        selected[(1, 1)](destination, torch.tensor([0]), torch.tensor([[2.]]),
                         1, 1, 1, 1, 1, BLOCK_SIZE_INDEX=1, BLOCK_SIZE_COL=512)
        original_launches += 1
        assert destination.item() == 2. and destination.item() != 7.
        # CPU interpreter submission order is deterministic here; the GPU
        # original has a write race for these repeated indices.
        destination = torch.full((2, 513), 5.)
        incoming = torch.stack((torch.full((513,), 2.), torch.full((513,), 3.)))
        selected[(2, 2)](destination, torch.tensor([0, 0]), incoming,
                         2, 2, 513, 513, 1, BLOCK_SIZE_INDEX=1, BLOCK_SIZE_COL=512)
        original_launches += 1
        assert torch.equal(destination[0], torch.full((513,), 3.))
        assert torch.equal(destination[1], torch.full((513,), 5.))

print(json.dumps({
    "mode": "actual Triton CPU interpreter; CUDA metadata/context facade; no GPU validation",
    "wrapper_cases": case_count,
    "empty_cases": empty_count,
    "rejections_before_launch": rejection_count,
    "repair_launches": Launch.count,
    "shared_read_storage_cases": 1,
    "original_pair_launches": original_launches,
    "candidate_ast_changes": changes,
    "original_seed_counterexample": {"initial": 5, "incoming": 2, "actual": 2, "target": 7},
    "passed": ["duplicates", "ordered FP32 accumulation", "nonzero seed", "tails", "offset views",
               "singleton strides", "raw unselected special words", "frame", "empty work",
               "guards", "read-role alias", "historical overwrite defect"]
}, sort_keys=True))
