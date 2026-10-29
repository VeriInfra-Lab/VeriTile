"""Actual Triton CPU interpreter checks for both rotary repair kernels.

These direct launches test addresses, masks, branch/store behavior and exact
integer-valued rotation cases. They do not test native GPU FMA conformance.
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interleaved = load("rotary_interpreter_interleaved", HERE / "rotary_emb_repaired.py")
nopad = load("rotary_interpreter_nopad", HERE.parent / "rotary_emb_nopad/rotary_emb_nopad_repaired.py")
torch.manual_seed(314159)
launches = []


def allocation(shape):
    size = 1
    for extent in shape:
        size *= extent
    backing = torch.randint(-4, 5, (size + 11,)).float()
    tensor = backing[:size].reshape(shape)
    return tensor, backing, backing.clone()


def bits_equal(left, right):
    assert torch.equal(left.view(torch.int32), right.view(torch.int32))


def interleaved_case(tokens, hq, hk, width, rotary, cw, sw):
    allocations = [allocation(shape) for shape in
                   [(tokens, hq, width), (tokens, hk, width), (tokens, cw), (tokens, sw)]]
    q, k, cosine, sine = [a[0] for a in allocations]
    expected = []
    for tensor in (q, k):
        result = tensor.clone()
        low, high = tensor[..., :rotary:2], tensor[..., 1:rotary:2]
        result[..., :rotary:2] = low * cosine[:, None, :rotary:2] - high * sine[:, None, :rotary:2]
        result[..., 1:rotary:2] = low * sine[:, None, 1:rotary:2] + high * cosine[:, None, 1:rotary:2]
        expected.append(result)
    grid = (triton.cdiv(max(hq, hk), 4), triton.cdiv(tokens, 16))
    interleaved._rotary_kernel[grid](
        q, k, cosine, sine, *q.stride(), *k.stride(), *cosine.stride(), *sine.stride(),
        tokens, rotary, hq, hk, BLOCK_HEAD=4, BLOCK_SEQ=16,
        BLOCK_PAIRS=triton.next_power_of_2(rotary // 2),
        num_warps=8 if rotary >= 128 else 4, num_stages=1)
    bits_equal(q, expected[0])
    bits_equal(k, expected[1])
    for tensor, backing, before in allocations:
        bits_equal(backing[tensor.numel():], before[tensor.numel():])
    for _, backing, before in allocations[2:]:
        bits_equal(backing, before)
    launches.append({"kernel": "interleaved", "grid": grid, "width": width,
                     "rotary": rotary, "hq": hq, "hk": hk})


def split_case(cached, tokens, hq, hk, width, cw, sw):
    allocations = [allocation(shape) for shape in
                   [(tokens, hq, width), (tokens, hk, width), (tokens, cw), (tokens, sw)]]
    q, k, cosine, sine = [a[0] for a in allocations]
    half = width // 2
    expected = []
    for tensor in (q, k):
        low, high = tensor[..., :half], tensor[..., half:]
        expected.append(torch.cat((low * cosine[:, None, :half] - high * sine[:, None, :half],
                                   low * sine[:, None, :half] + high * cosine[:, None, :half]), dim=-1))
    warps = 16 if width >= 512 else 8 if width >= 256 else 4
    if not cached:
        grid = (hq, triton.cdiv(tokens, 4))
        nopad.rotary_embedding_kernel[grid](
            q, k, cosine, sine, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            q.stride(2), *cosine.stride(), *sine.stride(), tokens,
            Q_HEAD_NUM=hq, KV_GROUP_NUM=hq // hk, HEAD_DIM=width,
            BLOCK_PAIRS=triton.next_power_of_2(half), BLOCK_TOKENS=4, num_warps=warps)
        bits_equal(k, expected[1])
    else:
        block_size = 4
        cache, cache_backing, cache_before = allocation((tokens + 2, hk, block_size, width))
        # Each token selects a separate physical block, with both nonzero
        # pages and slot remainders represented among the metadata.
        lengths = torch.tensor([1 + 5 * t for t in range(tokens)], dtype=torch.int64)
        columns = max(1, int((lengths.max().item() - 1) // block_size) + 1)
        tables = torch.zeros((tokens, columns), dtype=torch.int64)
        cache_expected = cache.clone()
        for token, length in enumerate(lengths.tolist()):
            page, slot = divmod(length - 1, block_size)
            tables[token, page] = token + 1
            cache_expected[token + 1, :, slot, :] = expected[1][token]
        tables_before, lengths_before = tables.clone(), lengths.clone()
        grid = (triton.next_power_of_2(hq), tokens)
        nopad.fused_rotary_embedding_kernel_v2[grid](
            q, k, cosine, sine, cache, tables, lengths,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), q.stride(2),
            *cosine.stride(), *sine.stride(), *cache.stride(), *tables.stride(),
            block_size, tokens, Q_HEAD_NUM=hq, K_HEAD_NUM=hk, HEAD_DIM=width,
            BLOCK_PAIRS=triton.next_power_of_2(half), num_warps=warps)
        bits_equal(cache, cache_expected)
        bits_equal(cache_backing[cache.numel():], cache_before[cache.numel():])
        bits_equal(allocations[1][1], allocations[1][2])
        assert torch.equal(tables, tables_before) and torch.equal(lengths, lengths_before)
    bits_equal(q, expected[0])
    for tensor, backing, before in allocations:
        bits_equal(backing[tensor.numel():], before[tensor.numel():])
    for _, backing, before in allocations[2:]:
        bits_equal(backing, before)
    launches.append({"kernel": "cached" if cached else "uncached", "grid": grid,
                     "width": width, "hq": hq, "hk": hk})


for config in [(1, 1, 1, 2, 2, 3, 5), (17, 3, 5, 6, 6, 8, 10),
               (2, 5, 1, 10, 6, 8, 12), (2, 1, 7, 258, 258, 260, 262)]:
    interleaved_case(*config)
for cached in (False, True):
    for config in [(5, 4, 2, 12, 8, 10), (2, 6, 2, 256, 129, 131), (1, 4, 1, 512, 258, 260)]:
        split_case(cached, *config)
print(json.dumps({"mode": "actual Triton CPU interpreter; direct kernel launches",
                  "gpu_execution": False, "gpu_primitive_conformance": False,
                  "launch_count": len(launches), "launches": launches,
                  "passed": ["independent coefficient strides", "interleaved and split-half formulas",
                             "partial rotary tail", "head/token/pair masks", "GQA leaders",
                             "cached K preservation", "page/slot metadata", "exterior frame",
                             "coefficient and metadata preservation"]}, sort_keys=True))
