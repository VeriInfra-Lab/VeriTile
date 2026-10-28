"""Python transcription of the existing Lean integer-address candidate.

Only the product in the first vector key load of each policy is commuted.
The original helper, all other statements, casts, masks and operations remain.
This file provides source correspondence, not a new runtime entrypoint.
"""
import triton
import triton.language as tl
from .kv_cache_filling_repaired import _quant_row

@triton.jit
def _fill_rows_commuted(K, V, KC, VC, KSZ, VSZ, SRC, DST,
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
        k = tl.load(K + DK * src + safe_ik, ik < DK, other=0.0)
        v = tl.load(V + src * DV + safe_iv, iv < DV, other=0.0)
        tl.store(KC + dst * DK + safe_ik, k, ik < DK)
        tl.store(VC + dst * DV + safe_iv, v, iv < DV)
    elif POLICY == 8:
        # Repeat a real lane in padding; masked undefined values never reduce.
        k0 = tl.load(K + src * DK)
        v0 = tl.load(V + src * DV)
        k = tl.load(K + DK * src + safe_ik, ik < DK, other=0.0)
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
        k1 = tl.load(K + DK * src + safe_ik, ik < CK, other=0.0).to(tl.float32)
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
