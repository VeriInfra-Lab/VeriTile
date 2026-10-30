"""Paged K/V filling with explicit metadata ownership and source precision.

State/cache tensors are native, contiguous, and use separate writable storage.
Sequence vectors and the page table are read by logical coordinates. Policy 0
copies FP32; policies 4/8 use uint8 caches and FP32 scale/zero metadata.
"""
from typing import Literal

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_row(x, lo, hi, LEVELS: tl.constexpr):
    raw_scale = (hi - lo) / LEVELS
    scale = tl.where(hi == lo, 1.0, raw_scale)
    zero = -lo / scale
    code = (x / scale + zero + 0.5).to(tl.uint8)
    return code, scale, zero


@triton.jit
def _fill_rows(K, V, KC, VC, KSZ, VSZ, SRC, DST,
               DK: tl.constexpr, DV: tl.constexpr,
               CK: tl.constexpr, CV: tl.constexpr,
               BK: tl.constexpr, BV: tl.constexpr, POLICY: tl.constexpr):
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, BK).to(tl.int64)
    iv = tl.arange(0, BV).to(tl.int64)
    safe_ik = tl.where(ik < CK, ik, 0)
    safe_iv = tl.where(iv < CV, iv, 0)
    if POLICY == 0:
        k = tl.load(K + src * DK + safe_ik, ik < DK, other=0.0)
        v = tl.load(V + src * DV + safe_iv, iv < DV, other=0.0)
        tl.store(KC + dst * DK + safe_ik, k, ik < DK)
        tl.store(VC + dst * DV + safe_iv, v, iv < DV)
    elif POLICY == 8:
        # Repeat a real lane in padding; masked undefined values never reduce.
        k0 = tl.load(K + src * DK)
        v0 = tl.load(V + src * DV)
        k = tl.load(K + src * DK + safe_ik, ik < DK, other=0.0)
        v = tl.load(V + src * DV + safe_iv, iv < DV, other=0.0)
        k = tl.where(ik < DK, k, k0)
        v = tl.where(iv < DV, v, v0)
        qk, sk, zk = _quant_row(k, tl.min(k, 0), tl.max(k, 0), 255)
        qv, sv, zv = _quant_row(v, tl.min(v, 0), tl.max(v, 0), 255)
        tl.store(KC + dst * CK + safe_ik, qk, ik < CK)
        tl.store(VC + dst * CV + safe_iv, qv, iv < CV)
        tl.store(KSZ + dst * 2, sk)
        tl.store(KSZ + dst * 2 + 1, zk)
        tl.store(VSZ + dst * 2, sv)
        tl.store(VSZ + dst * 2 + 1, zv)
    else:
        k10 = tl.load(K + src * DK)
        k20 = tl.load(K + src * DK + CK)
        v10 = tl.load(V + src * DV)
        v20 = tl.load(V + src * DV + CV)
        k1 = tl.load(K + src * DK + safe_ik, ik < CK, other=0.0).to(tl.float32)
        k2 = tl.load(K + src * DK + CK + safe_ik, ik < CK, other=0.0).to(tl.float32)
        v1 = tl.load(V + src * DV + safe_iv, iv < CV, other=0.0).to(tl.float32)
        v2 = tl.load(V + src * DV + CV + safe_iv, iv < CV, other=0.0).to(tl.float32)
        k1 = tl.where(ik < CK, k1, k10)
        k2 = tl.where(ik < CK, k2, k20)
        v1 = tl.where(iv < CV, v1, v10)
        v2 = tl.where(iv < CV, v2, v20)
        klo = tl.min(tl.minimum(k1, k2), 0)
        khi = tl.max(tl.maximum(k1, k2), 0)
        vlo = tl.min(tl.minimum(v1, v2), 0)
        vhi = tl.max(tl.maximum(v1, v2), 0)
        qk1, sk, zk = _quant_row(k1, klo, khi, 15)
        qk2, _, _ = _quant_row(k2, klo, khi, 15)
        qv1, sv, zv = _quant_row(v1, vlo, vhi, 15)
        qv2, _, _ = _quant_row(v2, vlo, vhi, 15)
        qk = qk1 + qk2 * 16
        qv = qv1 + qv2 * 16
        tl.store(KC + dst * CK + safe_ik, qk, ik < CK)
        tl.store(VC + dst * CV + safe_iv, qv, iv < CV)
        tl.store(KSZ + dst * 2, sk)
        tl.store(KSZ + dst * 2 + 1, zk)
        tl.store(VSZ + dst * 2, sv)
        tl.store(VSZ + dst * 2 + 1, zv)


def fill_kv_cache(k_states: torch.Tensor, v_states: torch.Tensor,
                  k_caches: torch.Tensor, v_caches: torch.Tensor,
                  q_start_loc: torch.Tensor, q_seq_length: torch.Tensor,
                  kv_seq_length: torch.Tensor, max_q_seq_length: int,
                  block_offsets: torch.Tensor,
                  k_scales_zeros: torch.Tensor = None,
                  v_scales_zeros: torch.Tensor = None,
                  quant_policy: Literal[0, 4, 8] = 0):
    """Validate metadata, fill each owned token/head, and return None.

The success contract assumes ordinary CUDA/Triton launch realization. All active tensors use the same CUDA device.
"""
    if quant_policy not in (0, 4, 8):
        raise ValueError('quant_policy must be 0, 4, or 8')
    if type(max_q_seq_length) is not int or max_q_seq_length < 0:
        raise ValueError('max_q_seq_length must be a nonnegative integer')
    tensors = [k_states, v_states, k_caches, v_caches,
               q_start_loc, q_seq_length, kv_seq_length, block_offsets]
    if quant_policy:
        tensors += [k_scales_zeros, v_scales_zeros]
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError('all active tensor roles are required')
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError('active tensors must have strided layout')
    if not k_states.is_cuda:
        raise ValueError('state/cache execution requires CUDA')
    if any(t.device != k_states.device for t in tensors):
        raise ValueError('active tensors must share one device')
    for t in [k_states, v_states, k_caches, v_caches] + (
            [k_scales_zeros, v_scales_zeros] if quant_policy else []):
        if not t.is_contiguous() or t.is_neg() or t.is_conj():
            raise ValueError('state/cache roles must be native contiguous tensors')
    if k_states.ndim != 3 or v_states.ndim != 3 or k_caches.ndim != 4 or v_caches.ndim != 4:
        raise ValueError('state rank is three and cache rank is four')
    tokens, heads, dk = k_states.shape
    pages, block, cache_heads, ck = k_caches.shape
    dv = v_states.shape[-1]
    cv = v_caches.shape[-1]
    factor = 2 if quant_policy == 4 else 1
    if (min(heads, dk, dv, block) <= 0 or cache_heads != heads
            or v_states.shape[:2] != (tokens, heads)
            or v_caches.shape[:3] != (pages, block, heads)
            or dk != factor * ck or dv != factor * cv):
        raise ValueError('incompatible state/cache dimensions')
    cache_dtype = torch.float32 if quant_policy == 0 else torch.uint8
    if k_states.dtype != torch.float32 or v_states.dtype != torch.float32:
        raise ValueError('state precision is FP32')
    if k_caches.dtype != cache_dtype or v_caches.dtype != cache_dtype:
        raise ValueError('cache dtype does not match the selected policy')
    if block_offsets.ndim != 2 or block_offsets.dtype != torch.int32:
        raise ValueError('page table must be rank-two INT32')
    batches = block_offsets.shape[0]
    for t in [q_start_loc, q_seq_length, kv_seq_length]:
        if t.dtype != torch.int32 or t.shape != (batches,):
            raise ValueError('sequence metadata must be rank-one INT32')
    if quant_policy:
        for t in [k_scales_zeros, v_scales_zeros]:
            if t.dtype != torch.float32 or t.shape != (pages, block, heads, 2):
                raise ValueError('scale/zero roles must have shape [pages,block,heads,2] in FP32')
    if max(ck, cv) > 2**20:
        raise ValueError('packed head width exceeds the supported block limit')
    payloads = [k_states, v_states, k_caches, v_caches] + ([k_scales_zeros, v_scales_zeros] if quant_policy else [])
    if any(t.numel() * t.element_size() >= 2**63 for t in payloads):
        raise ValueError('payload byte extents must fit signed int64')
    if any(t.numel() and t.data_ptr() % t.element_size() for t in payloads):
        raise ValueError('nonempty payload pointers must have natural alignment')
    outputs = [k_caches, v_caches] + ([k_scales_zeros, v_scales_zeros] if quant_policy else [])
    inputs = [k_states, v_states, q_start_loc, q_seq_length, kv_seq_length, block_offsets]
    def separate(a, b):
        sa, sb = a.untyped_storage(), b.untyped_storage()
        return (sa.nbytes() == 0 or sb.nbytes() == 0
                or sa.data_ptr() + sa.nbytes() <= sb.data_ptr()
                or sb.data_ptr() + sb.nbytes() <= sa.data_ptr())
    if any(not separate(a, b) for i, a in enumerate(outputs) for b in outputs[i+1:] + inputs):
        raise ValueError('writable allocations must be disjoint from other active roles')
    starts = q_start_loc.cpu().tolist()
    lengths = q_seq_length.cpu().tolist()
    totals = kv_seq_length.cpu().tolist()
    table = block_offsets.cpu().tolist()
    src_rows, dst_rows, occupied = [], [], set()
    for batch in range(batches):
        start, q, total = starts[batch], lengths[batch], totals[batch]
        if not (0 <= q <= max_q_seq_length and 0 <= start and start + q <= tokens and q <= total):
            raise ValueError('invalid source interval or sequence history')
        history = total - q
        for token in range(q):
            page_index, slot = divmod(history + token, block)
            if page_index >= len(table[batch]):
                raise ValueError('page table does not cover active tokens')
            page = table[batch][page_index]
            if not 0 <= page < pages:
                raise ValueError('active page outside cache allocation')
            dest_token = page * block + slot
            if dest_token in occupied:
                raise ValueError('two active tokens share a cache destination')
            occupied.add(dest_token)
            for head in range(heads):
                src_rows.append((start + token) * heads + head)
                dst_rows.append(dest_token * heads + head)
    if len(src_rows) >= 2**31:
        raise ValueError('job grid exceeds the native launch bound')
    if not src_rows:
        return None
    with torch.cuda.device(k_states.device):
        src = torch.tensor(src_rows, dtype=torch.int64, device=k_states.device)
        dst = torch.tensor(dst_rows, dtype=torch.int64, device=k_states.device)
        _fill_rows[(len(src_rows),)](
            k_states, v_states, k_caches, v_caches,
            k_scales_zeros if quant_policy else k_caches,
            v_scales_zeros if quant_policy else v_caches, src, dst,
            DK=dk, DV=dv, CK=ck, CV=cv,
            BK=triton.next_power_of_2(ck), BV=triton.next_power_of_2(cv),
            POLICY=quant_policy, num_warps=4, num_stages=3)
    return None
