"""Kernel-only source controls; CPU interpretation is not native FP validation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import types
import warnings

os.environ["TRITON_INTERPRET"] = "1"
sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)
DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = DIRECTORY.parents[3]
SELECTED = DIRECTORY / "cross_entropy2_repaired.py"
ORIGINAL = ROOT / "bench/tritonbench_g/cross_entropy2/cross_entropy2.py"


def load_kernels(path, name):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
             or (isinstance(node, ast.FunctionDef) and node.name.endswith("_kernel"))]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def backing_bytes(tensor):
    size = tensor.untyped_storage().nbytes()
    return torch.empty(0, dtype=torch.uint8).set_(tensor.untyped_storage(), 0, (size,), (1,))


def view(rows, cols, offset=3, row_gap=2):
    stride = cols + row_gap
    backing = torch.full((offset + rows * stride + 3,), 123.25)
    return backing.as_strided((rows, cols), (stride, 1), offset)


def vector(size, offset=2, stride=1, dtype=torch.float32):
    backing = torch.full((offset + size * stride + 3,), 91, dtype=dtype)
    return backing.as_strided((size,), (stride,), offset)


def footprint(tensor):
    if tensor.ndim == 1:
        cells = [tensor.storage_offset() + i * tensor.stride(0) for i in range(tensor.numel())]
    else:
        cells = [tensor.storage_offset() + r * tensor.stride(0) + c * tensor.stride(1)
                 for r in range(tensor.shape[0]) for c in range(tensor.shape[1])]
    return {tensor.element_size() * i + j for i in cells for j in range(tensor.element_size())}


def frame(tensor, before):
    after = backing_bytes(tensor)
    written = footprint(tensor)
    outside = [i for i in range(after.numel()) if i not in written]
    assert torch.equal(after[outside], before[outside]), "write escaped output view"


def close(actual, expected):
    torch.testing.assert_close(actual.double(), expected.double(), rtol=3e-5, atol=3e-5)


def case(module, cols, block, split, smoothing, enabled, scale, zscale, start=0, total=None, singleton=False, exceptional=False):
    total = total or cols
    rows = 1 if singleton else 4
    logits = view(rows, cols, row_gap=-cols if singleton else 2)
    logits.copy_(torch.arange(rows * cols).reshape(rows, cols).float() / 9 - 1)
    labels = vector(rows, dtype=torch.int32)
    ignored = -(1 << 31)
    labels.copy_(torch.tensor([min(start, (1 << 31) - 1)] if singleton else [start, start + cols - 1, 0, ignored], dtype=torch.int32))
    splits = (cols + block - 1) // block
    loss, lse, zloss = (vector(rows * splits, offset=i) for i in (2, 4, 6))
    tensors = (logits, labels, loss, lse, zloss)
    before = [backing_bytes(t).clone() for t in tensors]
    module.cross_entropy_fwd_kernel[(rows, splits)](
        loss, lse, zloss, logits, labels, smoothing, scale, zscale, ignored,
        total, start, cols, rows, logits.stride(0), BLOCK_SIZE=block,
        HAS_SMOOTHING=enabled, SPLIT=split,
    )
    x = logits.double() * scale
    smooth = smoothing if enabled else 0.0
    for b in range(splits):
        begin, end = b * block, min(cols, (b + 1) * block)
        local_lse = torch.logsumexp(x[:, begin:end], dim=1)
        expected = -smooth * x[:, begin:end].sum(dim=1) / total
        for r in range(rows):
            target = labels[r].item() - start
            if begin <= target < end:
                expected[r] -= (1 - smooth) * x[r, target]
        expected_z = torch.zeros_like(local_lse) if split else zscale * local_lse.square()
        if not split:
            expected += local_lse + expected_z
        expected = torch.where(labels == ignored, 0.0, expected)
        expected_z = torch.where(labels == ignored, 0.0, expected_z)
        close(lse.reshape(splits, rows)[b], local_lse)
        close(loss.reshape(splits, rows)[b], expected)
        close(zloss.reshape(splits, rows)[b], expected_z)
    for tensor, snapshot in zip(tensors[:2], before[:2]):
        assert torch.equal(backing_bytes(tensor), snapshot)
    for tensor, snapshot in zip(tensors[2:], before[2:]):
        frame(tensor, snapshot)
    supplied_lse = vector(rows)
    supplied_lse.copy_(torch.logsumexp(x, dim=1) + 0.125)
    upstream = vector(rows, stride=0 if singleton else 1)
    upstream.copy_(torch.tensor([-0.75] if singleton else [1.25, -0.75, 0.5, float("nan")]))
    if exceptional and not singleton:
        logits[-1].fill_(float("nan"))
        supplied_lse[-1] = float("inf")
    gradient = view(rows, cols, offset=4, row_gap=-cols if singleton else 3)
    x = logits.double() * scale
    expected = torch.exp(x - supplied_lse.double()[:, None])
    expected *= 1 + 2 * zscale * supplied_lse.double()[:, None]
    expected -= smooth / total
    for r in range(rows):
        target = labels[r].item() - start
        if 0 <= target < cols:
            expected[r, target] -= 1 - smooth
    expected *= upstream.double()[:, None] * scale
    expected[labels == ignored] = 0.0
    inputs = (logits, labels, supplied_lse, upstream)
    snapshots = [backing_bytes(t).clone() for t in inputs]
    out_before = backing_bytes(gradient).clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module.cross_entropy_bwd_kernel[(rows, splits)](
            gradient, upstream, logits, supplied_lse, labels, smoothing, scale, zscale,
            ignored, total, start, cols, logits.stride(0), gradient.stride(0), upstream.stride(0),
            BLOCK_SIZE=block, HAS_SMOOTHING=enabled,
        )
    close(gradient, expected)
    if not singleton:
        assert torch.equal(gradient[-1].view(torch.int32), torch.zeros(cols, dtype=torch.int32))
    frame(gradient, out_before)
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(backing_bytes(tensor), snapshot)
    if singleton:
        assert logits.is_contiguous() and logits.stride() == (0, 1)
        assert gradient.is_contiguous() and gradient.stride() == (0, 1)


def original_counterexamples(module):
    x = torch.tensor([[1.0, 2.0, 3.0]])
    labels = torch.tensor([0], dtype=torch.int32)
    loss, lse, zloss = (torch.full((1,), 77.0) for _ in range(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module.cross_entropy_fwd_kernel[(1, 1)](
            loss, lse, zloss, x, labels, 0.0, -1.0, 0.0, -100,
            3, 0, 3, 1, 3, BLOCK_SIZE=4, HAS_SMOOTHING=False, SPLIT=True,
        )
    assert torch.isnan(lse).all(), "original negative-scale tail defect changed"
    assert zloss.item() == 77.0, "original missing split z-loss store changed"


def main():
    selected = load_kernels(SELECTED, "ce2_selected_kernel_control")
    original = load_kernels(ORIGINAL, "ce2_original_kernel_control")
    cases = []
    for cols, block, split in [(1, 1, False), (3, 4, False), (5, 4, True)]:
        for scale in (-1.25, 0.0, 1.5):
            case(selected, cols, block, split, 0.25, True, scale, 0.125, exceptional=True)
            cases.append(f"C{cols}_B{block}_split{split}_scale{scale}_ignored_nan")
    for split in (False, True):
        for enabled in (False, True):
            case(selected, 3, 4, split, 0.25, enabled, -0.75, 0.125, start=5, total=13)
            cases.append(f"partition_nonowner_split{split}_smoothing_enabled{enabled}")
    for scale in (-0.0, -1.25):
        case(selected, 3, 4, False, -0.0, False, scale, -0.0, singleton=True)
        cases.append(f"singleton_stride0_scale{scale}_runtime_negative_zero")
    case(selected, 3, 4, True, 0.25, True, 1.0, 0.0, start=(1 << 63) - 5, total=(1 << 63) - 1, singleton=True)
    cases.append("int64_near_max_metadata_nonowner")
    original_counterexamples(original)
    print(json.dumps({
        "selected_source_sha256": hashlib.sha256(SELECTED.read_bytes()).hexdigest(),
        "original_source_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        "kernel_only_cpu_controls": cases,
        "original_counterexamples": ["negative_scale_masked_tail_nan", "missing_split_z_loss_store"],
        "full_output_backing_frames": "passed",
        "gpu_execution": False,
        "native_numerical_equivalence": "not_claimed_by_cpu_controls",
    }, indent=2))


if __name__ == "__main__":
    main()
