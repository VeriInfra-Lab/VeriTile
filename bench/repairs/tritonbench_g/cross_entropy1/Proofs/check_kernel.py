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
SELECTED = DIRECTORY / "cross_entropy1_repaired.py"
ORIGINAL = ROOT / "bench/tritonbench_g/cross_entropy1/cross_entropy1.py"


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


def forward_case(module, cols, block, split, smoothing, scale):
    rows = 3
    logits = view(rows, cols)
    logits.copy_(torch.arange(rows * cols).reshape(rows, cols).float() / 7 - 1)
    labels = vector(rows, dtype=torch.int64)
    labels.copy_(torch.tensor([0, cols - 1, -100]))
    splits = (cols + block - 1) // block
    losses = vector(rows * splits)
    lses = vector(rows * splits, offset=4)
    if splits > 1:
        losses, lses = losses.view(splits, rows), lses.view(splits, rows)
    input_frames = [backing_bytes(t).clone() for t in (logits, labels)]
    output_frames = [backing_bytes(t).clone() for t in (losses, lses)]
    module.cross_entropy_fwd_kernel[(rows, splits)](
        losses, lses, logits, labels, smoothing, scale, -100, cols, 0,
        cols, rows, logits.stride(0), BLOCK_SIZE=block, SPLIT=split,
    )
    x = logits.double()
    for b in range(splits):
        begin, end = b * block, min(cols, (b + 1) * block)
        local_lse = torch.logsumexp(x[:, begin:end], dim=1)
        local_sum = x[:, begin:end].sum(dim=1)
        expected = -smoothing * local_sum / cols
        for r in range(rows):
            target = labels[r].item()
            if begin <= target < end:
                expected[r] -= (1 - smoothing) * x[r, target]
        if not split:
            expected += local_lse + scale * local_lse.square()
        expected = torch.where(labels == -100, 0.0, expected)
        close(lses.reshape(splits, rows)[b], local_lse)
        close(losses.reshape(splits, rows)[b], expected)
    for tensor, before in zip((logits, labels), input_frames):
        assert torch.equal(backing_bytes(tensor), before)
    for tensor, before in zip((losses, lses), output_frames):
        frame(tensor, before)


def backward_case(module, cols, block, smoothing, scale, inplace=False, exceptional=False):
    rows = 3
    logits = view(rows, cols)
    logits.copy_(torch.arange(rows * cols).reshape(rows, cols).float() / 9 - 1)
    labels = vector(rows, dtype=torch.int64)
    labels.copy_(torch.tensor([0, cols - 1, -(1 << 63)]))
    lse = vector(rows)
    lse.copy_(torch.logsumexp(logits.double(), dim=1))
    upstream = vector(rows, stride=2)
    upstream.copy_(torch.tensor([1.25, -0.75, float("nan")]))
    if exceptional:
        logits[2].fill_(float("nan"))
        lse[2] = float("inf")
    gradient = logits if inplace else view(rows, cols, offset=4, row_gap=3)
    x = logits.double().clone()
    expected = ((1 + 2 * scale * lse.double()[:, None]) *
                torch.exp(x - lse.double()[:, None]) - smoothing / cols)
    expected[0, 0] -= 1 - smoothing
    expected[1, cols - 1] -= 1 - smoothing
    expected *= upstream.double()[:, None]
    expected[2].zero_()
    tensors = (logits, labels, lse, upstream)
    before = [backing_bytes(t).clone() for t in tensors]
    out_before = backing_bytes(gradient).clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module.cross_entropy_bwd_kernel[(rows, (cols + block - 1) // block)](
            gradient, upstream, logits, lse, labels, smoothing, scale, -(1 << 63),
            cols, 0, cols, logits.stride(0), gradient.stride(0), upstream.stride(0),
            BLOCK_SIZE=block,
        )
    close(gradient, expected)
    assert torch.equal(gradient[2].view(torch.int32), torch.zeros(cols, dtype=torch.int32))
    frame(gradient, out_before)
    for tensor, snapshot in zip(tensors, before):
        if tensor is not gradient:
            assert torch.equal(backing_bytes(tensor), snapshot)


def original_ignored_counterexample(module):
    logits = torch.tensor([[float("nan"), 1.]])
    labels = torch.tensor([-100], dtype=torch.int64)
    lse = torch.tensor([float("nan")])
    upstream = torch.tensor([1.])
    gradient = torch.full_like(logits, 77.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module.cross_entropy_bwd_kernel[(1, 1)](
            gradient, upstream, logits, lse, labels, 0., 0., -100, 2, 0, 2,
            2, 2, 1, BLOCK_SIZE=2,
        )
    assert torch.isnan(gradient).all(), "original ignored-row counterexample changed"


def singleton_stride_zero_case(module, smoothing):
    """The wrapper admits singleton contiguous rows with an arbitrary row stride."""
    cols, block = 3, 4
    scale = -0.0
    assert struct.pack("<f", scale) == b"\x00\x00\x00\x80"
    if smoothing == 0.0:
        assert struct.pack("<f", smoothing) == b"\x00\x00\x00\x80"
    logits = view(1, cols, row_gap=-cols)
    logits.copy_(torch.tensor([[-1.25, 0.5, 2.0]]))
    assert logits.is_contiguous() and logits.stride() == (0, 1)
    labels = vector(1, dtype=torch.int64)
    labels[0] = 1
    lse, loss = vector(1), vector(1, offset=4)
    tensors = (logits, labels, lse, loss)
    before = [backing_bytes(t).clone() for t in tensors]
    x = logits.double().clone()
    expected_lse = torch.logsumexp(x, dim=1)
    expected_loss = expected_lse - (1 - smoothing) * x[:, 1] - smoothing * x.mean(dim=1)
    module.cross_entropy_fwd_kernel[(1, 1)](
        loss, lse, logits, labels, smoothing, scale, -100, cols, 0,
        cols, 1, 0, BLOCK_SIZE=block, SPLIT=False,
    )
    close(lse, expected_lse)
    close(loss, expected_loss)
    for tensor, snapshot in zip(tensors[:2], before[:2]):
        assert torch.equal(backing_bytes(tensor), snapshot)
    for tensor, snapshot in zip(tensors[2:], before[2:]):
        frame(tensor, snapshot)
    upstream = vector(1, stride=0)
    upstream[0] = -0.75
    preserved = [backing_bytes(t).clone() for t in (labels, lse, upstream)]
    expected_gradient = torch.exp(x - lse.double()[:, None]) - smoothing / cols
    expected_gradient[0, 1] -= 1 - smoothing
    expected_gradient *= upstream.double()[:, None]
    module.cross_entropy_bwd_kernel[(1, 1)](
        logits, upstream, logits, lse, labels, smoothing, scale, -100,
        cols, 0, cols, 0, 0, 0, BLOCK_SIZE=block,
    )
    close(logits, expected_gradient)
    frame(logits, before[0])
    for tensor, snapshot in zip((labels, lse, upstream), preserved):
        assert torch.equal(backing_bytes(tensor), snapshot)


def main():
    selected = load_kernels(SELECTED, "ce1_selected_kernel_control")
    original = load_kernels(ORIGINAL, "ce1_original_kernel_control")
    cases = []
    for cols, block, split in [(1, 1, False), (3, 4, False), (5, 4, True)]:
        for smoothing in (0., .25):
            forward_case(selected, cols, block, split, smoothing, .125)
            cases.append(f"forward_C{cols}_B{block}_split{split}_smooth{smoothing}")
    for smoothing in (0., .25):
        for inplace in (False, True):
            backward_case(selected, 5, 4, smoothing, .125, inplace, exceptional=True)
            cases.append(f"backward_tail_smooth{smoothing}_inplace{inplace}_ignored_nan_int64_min")
    for smoothing in (-0.0, 0.25):
        singleton_stride_zero_case(selected, smoothing)
        cases.append(f"forward_backward_singleton_stride0_inplace_smooth{smoothing}_scale_negative_zero")
    original_ignored_counterexample(original)
    print(json.dumps({
        "selected_source_sha256": hashlib.sha256(SELECTED.read_bytes()).hexdigest(),
        "original_source_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        "kernel_only_cpu_controls": cases,
        "original_ignored_nan_counterexample": "reproduced",
        "full_output_backing_frames": "passed",
        "gpu_execution": False,
        "native_numerical_equivalence": "not_claimed",
    }, indent=2))


if __name__ == "__main__":
    main()
