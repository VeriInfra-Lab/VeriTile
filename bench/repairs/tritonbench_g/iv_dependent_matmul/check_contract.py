"""Checked wrapper and actual Triton interpreter controls; no GPU execution."""
import os
os.environ["TRITON_INTERPRET"] = "1"
import ast
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "iv_dependent_matmul_i64.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/iv_dependent_matmul/iv_dependent_matmul.py"
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == "dfd772c19ff1b007b3e6c474fa984a460d0e135593cc3d0f05d279c8746997d7"
spec = importlib.util.spec_from_file_location("iv_checked", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MODES = ("pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters")
real_kernel = module.iv_dependent_matmul_kernel_i64
rand, empty = torch.rand, torch.empty
counts = {"reject": 0, "calls": 0, "launches": 0}

# Kernel arithmetic, casts, staging statements, masks and store remain pinned.
class EraseCasts(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "cast":
            return node.args[0]
        return node
    def visit_Name(self, node):
        if node.id == "k64":
            node.id = "k"
        return node

def kernel_ast(text, name):
    f = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    f.name = "kernel"
    f.decorator_list = []
    f = EraseCasts().visit(f)
    f.body = [n for n in f.body if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Name) and n.targets[0].id == n.value.id)]
    for n in ast.walk(f):
        if isinstance(n, ast.For):
            n.body = [x for x in n.body if not (isinstance(x, ast.Assign) and len(x.targets) == 1 and isinstance(x.targets[0], ast.Name) and isinstance(x.value, ast.Name) and x.targets[0].id == x.value.id)]
    return ast.dump(f, include_attributes=False)
assert kernel_ast(SOURCE.read_text(), "iv_dependent_matmul_kernel_i64") == kernel_ast(ORIGINAL.read_text(), "iv_dependent_matmul_kernel")

class Launcher:
    def __init__(self, allocated):
        self.allocated = allocated
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            counts["launches"] += 1
            a, b, c, M, N, K = args[:6]
            assert (a, b, c) == tuple(self.allocated)
            assert kwargs["num_stages"] == (4 if kwargs["type"] == "post_load_three_iters" else 3)
            assert grid(kwargs) == ((M + kwargs["BLOCK_SIZE_M"] - 1) // kwargs["BLOCK_SIZE_M"] * ((N + kwargs["BLOCK_SIZE_N"] - 1) // kwargs["BLOCK_SIZE_N"]),)
            before_a, before_b = a.clone(), b.clone()
            real_kernel[grid](*args, **kwargs)
            assert torch.equal(a, before_a) and torch.equal(b, before_b)
            expected = (a @ b).half().float()
            assert torch.equal(c, expected), (kwargs["type"], (M, K, N), (c - expected).abs().max())
        return launch

def successful(shape, mode, blocks=(32, 32, 32)):
    allocated = []
    backing = []
    def random(shape, *, device):
        assert device.type == "cuda"
        result = torch.randint(0, 3, shape, dtype=torch.int64).float() / 4
        allocated.append(result)
        return result
    def allocate(shape, *, device):
        assert device.type == "cuda"
        storage = empty((shape[0] * shape[1] + 8,), dtype=torch.float32).fill_(1234)
        backing.append(storage)
        result = storage[4:-4].view(shape)
        allocated.append(result)
        return result
    before = counts["launches"]
    with patch.object(torch, "rand", random), patch.object(torch, "empty", allocate), patch.object(module, "iv_dependent_matmul_kernel_i64", Launcher(allocated)):
        out = module.iv_dependent_matmul_wrapper_i64(*shape, *blocks, type=mode)
    assert out is allocated[2]
    assert torch.all(backing[0][:4] == 1234) and torch.all(backing[0][-4:] == 1234)
    assert out.shape == (shape[0], shape[2]) and out.dtype == torch.float32
    assert counts["launches"] - before == int(shape[0] != 0 and shape[2] != 0)
    counts["calls"] += 1

for mode in MODES:
    for shape in [(1, 0, 1), (1, 1, 1), (35, 65, 37), (33, 97, 17), (0, 17, 4), (4, 17, 0)]:
        successful(shape, mode)

base = dict(M=2, K=3, N=4, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32)
invalid = [dict(M=-1), dict(K=-1), dict(N=-1), dict(M=1.5), dict(K=True), dict(BLOCK_SIZE_M=False), dict(BLOCK_SIZE_M=0), dict(BLOCK_SIZE_N=3), dict(BLOCK_SIZE_K=2**21), dict(type="unknown"), dict(device="cpu"), dict(M=2**63), dict(K=2**63-1), dict(N=2**63-1), dict(M=2**31, K=2**31), dict(M=32*2**31, K=0, N=1)]
for changes in invalid:
    calls = []
    with patch.object(torch, "rand", lambda *a, **kw: calls.append("random")), patch.object(torch, "empty", lambda *a, **kw: calls.append("empty")):
        try:
            module.iv_dependent_matmul_wrapper_i64(**(base | changes))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(changes)
    assert not calls, changes
    counts["reject"] += 1
with patch.object(torch, "get_default_dtype", lambda: torch.float64), patch.object(torch, "rand", side_effect=AssertionError("allocation before rejection")):
    try:
        module.iv_dependent_matmul_wrapper_i64(**base)
    except ValueError:
        counts["reject"] += 1
    else:
        raise AssertionError("default dtype accepted")
print(counts, "all passed; CPU interpreter only")
