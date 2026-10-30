"""Guarded contiguous FP32 uniform sampling with atomic Philox reservations.

Source: uniform_sampling.py, SHA256
d2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1.
See README.md for state, numerical, runtime, and failure contracts.
"""
import functools
import operator
from pathlib import Path

import torch
import triton
import triton.language as tl


try:
    uint_to_uniform_float = tl.uint_to_uniform_float
except AttributeError:
    @triton.jit
    def uint_to_uniform_float(x):
        if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
            x = x.to(tl.int32, bitcast=True)
            scale = 4.6566127342e-10
        else:
            tl.static_assert(
                tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
            )
            x = x.to(tl.int64, bitcast=True)
            scale = 1.0842020432385337e-19
        x = tl.where(x < 0, -x - 1, x)
        return x * scale


def heur_block(args):
    return 512 if args["N"] <= 512 else 1024


def heur_num_warps(args):
    return 4 if args["N"] <= 512 else (8 if args["N"] <= 1024 else 16)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_sampling_repaired_kernel(
    out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    i4 = pid * BLOCK + lane
    c0 += i4.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    # Explicitly retain the default native affine FMA recorded for the source.
    r0 = tl.fma(uint_to_uniform_float(r0), to - from_, from_)
    r1 = tl.fma(uint_to_uniform_float(r1), to - from_, from_)
    r2 = tl.fma(uint_to_uniform_float(r2), to - from_, from_)
    r3 = tl.fma(uint_to_uniform_float(r3), to - from_, from_)
    off_0 = pid * BLOCK * 4 + lane
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + tl.minimum(off_0, N - 1), r0, mask=off_0 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_1, N - 1), r1, mask=off_1 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_2, N - 1), r2, mask=off_2 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_3, N - 1), r3, mask=off_3 < N,
             eviction_policy="evict_first")



@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_sampling_index_commuted_kernel(
    out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    i4 = BLOCK * pid + lane
    c0 += i4.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    # Explicitly retain the default native affine FMA recorded for the source.
    r0 = tl.fma(uint_to_uniform_float(r0), to - from_, from_)
    r1 = tl.fma(uint_to_uniform_float(r1), to - from_, from_)
    r2 = tl.fma(uint_to_uniform_float(r2), to - from_, from_)
    r3 = tl.fma(uint_to_uniform_float(r3), to - from_, from_)
    off_0 = pid * BLOCK * 4 + lane
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + tl.minimum(off_0, N - 1), r0, mask=off_0 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_1, N - 1), r1, mask=off_1 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_2, N - 1), r2, mask=off_2 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_3, N - 1), r3, mask=off_3 < N,
             eviction_policy="evict_first")


def volume(shape):
    return functools.reduce(operator.mul, shape, 1)


@functools.lru_cache(maxsize=1)
def _reservation_module():
    from torch.utils.cpp_extension import load
    return load(
        name="veritile_uniform_sampling_reservation_v1",
        sources=[str(Path(__file__).with_name("reserve_philox.cpp"))],
        extra_cflags=["-O2"], with_cuda=False, verbose=False,
    )


def _signed64(word):
    return word if word < (1 << 63) else word - (1 << 64)


def _launch_chunks(n, block, start):
    """Split only at low-word carry boundaries; retain global four-stream order."""
    done = 0
    counters = (n + 4 * block - 1) // (4 * block) * block
    while done < counters:
        offset = start + done
        available = (1 << 32) - (offset & 0xFFFFFFFF)
        count = min(counters - done, available)
        elements = min(n - 4 * done, 4 * count)
        yield 4 * done, elements, count // block, _signed64(offset)
        done += count


def uniform_sampling_repaired(self, from_=0.0, to=1.0, *, generator=None):
    if not isinstance(self, torch.Tensor) or self.layout != torch.strided:
        raise TypeError("uniform output must be a strided torch.Tensor")
    if self.dtype != torch.float32:
        raise TypeError("this repaired precision profile requires FP32 output")
    if self.device.type != "cuda":
        raise ValueError("uniform output must be on a CUDA device")
    if generator is not None and (
        not isinstance(generator, torch.Generator) or generator.device != self.device
    ):
        raise ValueError("generator must belong to the output CUDA device")
    if not self.is_contiguous():
        raise ValueError("uniform output must be contiguous")
    if self.is_neg():
        raise ValueError("uniform output must have no unresolved negative view bit")
    if self.numel() > 0 and self.data_ptr() % 4 != 0:
        raise ValueError("nonempty output must have a four-byte-aligned data pointer")
    if not isinstance(from_, float) or not isinstance(to, float):
        raise TypeError("uniform bounds must be Python floats")
    lower, upper = float(from_), float(to)
    n = volume(self.shape)
    if n >= 2**63:
        raise ValueError("uniform output extent must fit signed64")
    if n == 0:
        return self
    block = heur_block({"N": n})
    warps = heur_num_warps({"N": n})
    counter_count = (n + 4 * block - 1) // (4 * block) * block
    reservation = _reservation_module()
    with torch.cuda.device(self.device):
        scratch = torch.empty((n,), device=self.device, dtype=torch.float32)
        selected = (torch.cuda.default_generators[self.device.index]
                    if generator is None else generator)
        seed, start = reservation.reserve(selected, self.device.index, block, counter_count)
        for base, size, programs, offset in _launch_chunks(n, block, start):
            uniform_sampling_repaired_kernel[(programs,)](
                scratch[base:base + size], size, _signed64(seed), offset, lower, upper,
                BLOCK=block, num_warps=warps, enable_fp_fusion=True,
            )
        self.copy_(scratch.reshape(self.shape))
    return self
