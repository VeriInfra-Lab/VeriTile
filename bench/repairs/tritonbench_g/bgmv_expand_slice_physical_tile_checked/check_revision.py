"""Directed physical-alias/static-tile controls for this guard-only revision.

Real Torch storage and DLPack operations, with the existing CPU CUDA descriptor
adapter. No GPU execution or native numerical-conformance claim.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import ast
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "bgmv_expand_slice"
SOURCE = HERE / "bgmv_expand_slice_physical_tile_checked.py"
OUT = ROOT / ".veritile-cache/kernel-evidence/bgmv_expand_slice_physical_tile_checked"

import torch
import triton
from triton._utils import TRITON_MAX_TENSOR_NUMEL
torch.set_num_threads(1)
assert TRITON_MAX_TENSOR_NUMEL == 2**20


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


adapter = module("bgmv_physical_adapter", OLD / "check_contract.py")
candidate = module("bgmv_physical_candidate", SOURCE)
kernel, pids = adapter.kernel_ast(SOURCE)

old_tree = ast.parse((OLD / "bgmv_expand_slice_repaired.py").read_text())
new_tree = ast.parse(SOURCE.read_text())
old_kernel = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef))
new_kernel = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef))
assert ast.dump(old_kernel) == ast.dump(new_kernel)
old_wrapper = old_tree.body[-1]
new_wrapper = new_tree.body[-1]
old_wrapper.body = [n for n in old_wrapper.body if "_cdata" not in ast.unparse(n)]
new_wrapper.body = [n for n in new_wrapper.body
                    if not (isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id in
                        {"output_storage", "output_begin", "output_end"} for t in n.targets))
                    and not (isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                             and n.target.id == "operand")
                    and not (isinstance(n, ast.If) and "256 * BLOCK_K" in ast.unparse(n.test))]
assert ast.dump(old_wrapper) == ast.dump(new_wrapper)

context = [2]


@contextmanager
def device(value):
    saved = context[0]
    context[0] = value.index
    try:
        yield
    finally:
        context[0] = saved


class Dispatch:
    def __init__(self, execute):
        self.calls = []
        self.execute = execute

    def __getitem__(self, grid):
        def launch(*args, **kw):
            assert context[0] == args[0].device.index
            self.calls.append((grid, args, kw))
            if self.execute:
                values = [adapter.Pointer(a.data) if isinstance(a, adapter.Tensor) else a for a in args]
                for b in range(grid[1]):
                    for n in range(grid[0]):
                        pids[:] = [n, b]
                        kernel(*values, **kw)
        return launch


candidate.torch = SimpleNamespace(Tensor=adapter.Tensor, strided=torch.strided,
    float16=torch.float16, float32=torch.float32, int32=torch.int32, int64=torch.int64,
    all=torch.all, cuda=SimpleNamespace(device=device))
results = []


def call(name, tensors, *, add=False, offset=0, execute=False, reject=None, launches=1):
    before = [adapter.raw(t).clone() for t in tensors]
    dispatch = Dispatch(execute)
    candidate._bgmv_expand_slice_kernel = dispatch
    error = None
    try:
        returned = candidate._bgmv_expand_slice(*[adapter.Tensor(t) for t in tensors],
                                                offset, tensors[1].shape[-2], add)
        assert returned is None
    except ValueError as exc:
        error = str(exc)
    assert context[0] == 2
    if reject:
        assert error is not None and reject in error, (name, error)
        assert not dispatch.calls
    else:
        assert error is None, (name, error)
        assert len(dispatch.calls) == launches
    if reject or not execute:
        for t, saved in zip(tensors, before):
            assert torch.equal(adapter.raw(t), saved)
    if execute:
        for j in (0, 1, 3):
            assert torch.equal(adapter.raw(tensors[j]), before[j])
        expected = tensors[2].clone()
        # Restore entry Y values without changing its caller storage.
        old_y = tensors[2].clone()
        old_y.view(torch.uint8).reshape(-1).copy_(before[2][
            tensors[2].storage_offset()*4:tensors[2].storage_offset()*4+tensors[2].numel()*4])
        expected.copy_(old_y)
        weights = tensors[1].squeeze(1) if tensors[1].ndim == 4 else tensors[1]
        for b, slot in enumerate(tensors[3].tolist()):
            if slot == -1:
                continue
            value = (tensors[0][b].half()[None, :] * weights[slot]).half().sum(1, dtype=torch.float16).float()
            if add:
                value = value + old_y[b, offset:offset+weights.shape[-2]]
            expected[b, offset:offset+weights.shape[-2]] = value
        assert torch.equal(tensors[2], expected)
        changed = adapter.raw(tensors[2]) != before[2]
        allowed = torch.zeros_like(changed)
        for b, slot in enumerate(tensors[3].tolist()):
            if slot != -1:
                start = 4*(tensors[2].storage_offset()+b*tensors[2].shape[1]+offset)
                allowed[start:start+4*weights.shape[-2]] = True
        assert not torch.any(changed & ~allowed)
    row = {"name": name, "status": "rejected" if reject else "admitted", "reason": error,
           "launches": len(dispatch.calls), "complete_CPU_source_execution": execute,
           "input_backings_preserved": True, "output_complement_preserved": execute}
    if dispatch.calls:
        row["grid"] = list(dispatch.calls[0][0])
        row["constexprs"] = dispatch.calls[0][2]
    results.append(row)
    return dispatch


for role in ["input", "weight", "metadata_int32", "metadata_int64"]:
    if role == "input":
        x = torch.tensor([[1., 2.]])
        w = torch.tensor([[[2., 0.], [1., 0.]]], dtype=torch.float16)
        i = torch.tensor([0], dtype=torch.int32)
        y = torch.from_dlpack(x)
        shared = x
    elif role == "weight":
        x = torch.ones(1, 2)
        w = torch.tensor([[[1., 2.]]], dtype=torch.float16)
        i = torch.tensor([0], dtype=torch.int32)
        y = torch.from_dlpack(w).view(torch.float32).view(1, 1)
        shared = w
    else:
        x = torch.ones(1, 1)
        w = torch.tensor([[[2.]]], dtype=torch.float16)
        i = torch.tensor([0], dtype=torch.int64 if role.endswith("64") else torch.int32)
        y = torch.from_dlpack(i).view(torch.float32).view(1, -1)
        shared = i
    assert y.untyped_storage()._cdata != shared.untyped_storage()._cdata
    assert y.data_ptr() == shared.data_ptr()
    call("dlpack_"+role, [x, w, y, i], reject="backing byte ranges")

for k in [4096, 4097]:
    for four, wide, add in [(False, False, False), (True, True, True)]:
        tensors = [torch.ones(1, k), torch.ones((1, 1, 2, k) if four else (1, 2, k), dtype=torch.float16),
                   torch.zeros(1, 2), torch.tensor([0], dtype=torch.int64 if wide else torch.int32)]
        call(f"tile_K{k}_rank{4 if four else 3}", tensors, add=add,
             reject="static tensor element limit" if k == 4097 else None)

for b, n in [(0, 2), (1, 0)]:
    for k in [4097, 0]:
        tensors = [torch.ones(b, k), torch.ones(1, n, k, dtype=torch.float16), torch.zeros(b, n),
                   torch.full((b,), -2, dtype=torch.int64)]
        call(f"empty_B{b}_N{n}_K{k}", tensors, launches=0,
             reject="dimensions and output slice" if k == 0 else None)

# A complete run for each unchanged add/rank/index-width branch. The second
# run has real overlapping read backings and an aligned output offset view.
call("disjoint_rank3_i32_add0", [torch.tensor([[1., 2.]]),
     torch.tensor([[[2., 0.], [1., 0.]]], dtype=torch.float16), torch.zeros(1, 2),
     torch.tensor([0], dtype=torch.int32)], execute=True)
shared = torch.tensor([1., 2., 3., 4.])
x = shared[:2].view(1, 2)
w = torch.from_dlpack(shared).view(torch.float16)[:4].view(1, 1, 2, 2)
y = torch.full((12,), 3.)[4:8].view(1, 4)
assert x.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
call("read_alias_rank4_i64_add1_offset", [x, w, y, torch.tensor([0], dtype=torch.int64)],
     add=True, offset=1, execute=True)

report = {"status": "PASS", "source_sha256": digest(SOURCE),
          "historical_source_sha256": digest(OLD / "bgmv_expand_slice_repaired.py"),
          "checker_sha256": digest(Path(__file__)), "kernel_AST_unchanged": True,
          "wrapper_only_physical_separation_and_static_tile_checks_changed": True,
          "static_tensor_element_limit": TRITON_MAX_TENSOR_NUMEL,
          "controls": results, "historical_suite_rerun": False,
          "GPU_execution": False, "native_numerical_conformance": False}
(OUT / "revision-controls.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({"status": "PASS", "controls": len(results),
                  "rejected": sum(r["status"] == "rejected" for r in results),
                  "complete_CPU_source_executions": 2}, indent=2))
