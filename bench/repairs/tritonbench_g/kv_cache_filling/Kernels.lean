import bench.repairs.tritonbench_g.kv_cache_filling.Model
import VeriTile.Triton.DSL.CallSignature

/-! Complete original and pre-existing integer-address candidate bodies.
Caller types, constexpr values and helper definitions are supplied outside the
quotations. Policy branches, helper calls, casts and store spelling stay raw. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def quantRow : SourceFunction := triton_function _quant_row(x, lo, hi, LEVELS) constexpr [LEVELS] in triton {
    raw_scale = (hi - lo) / LEVELS
    scale = tl.where(hi == lo, 1.0, raw_scale)
    zero = -lo / scale
    code = (x / scale + zero + 0.5).to(tl.uint8)
    return code, scale, zero
}

namespace CopySource
structure Parameters where
  (dk dv ck cv bk bv : Nat)

def copyCall : CallSignature where
  pointers := [("K",.fp32),("V",.fp32),("KC",.fp32),("VC",.fp32),
    ("KSZ",.fp32),("VSZ",.fp32),("SRC",.i64),("DST",.i64)]
  scalars := []
  constexpr := ["DK","DV","CK","CV","BK","BV","POLICY"]
  functions := [quantRow]

def originalKernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyCache"
  let VSZ : RegionName := "valueCache"
  let SRC : RegionName := "src"
  let DST : RegionName := "dst"
  let DK := p.dk
  let DV := p.dv
  let CK := p.ck
  let CV := p.cv
  let BK := p.bk
  let BV := p.bv
  let POLICY : Nat := 0
  with_triton_signature copyCall in triton {
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
  }

def commutedKernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyCache"
  let VSZ : RegionName := "valueCache"
  let SRC : RegionName := "src"
  let DST : RegionName := "dst"
  let DK := p.dk
  let DV := p.dv
  let CK := p.ck
  let CV := p.cv
  let BK := p.bk
  let BV := p.bv
  let POLICY : Nat := 0
  with_triton_signature copyCall in triton {
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
  }

def kernel (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commutedKernel p else originalKernel p

end CopySource

namespace QuantSource
abbrev Parameters := CopySource.Parameters

def quant8Call : CallSignature where
  pointers := [("K",.fp32),("V",.fp32),("KC",.u8),("VC",.u8),
    ("KSZ",.fp32),("VSZ",.fp32),("SRC",.i64),("DST",.i64)]
  scalars := []
  constexpr := ["DK","DV","CK","CV","BK","BV","POLICY"]
  functions := [quantRow]

def original8Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : RegionName := "src"
  let DST : RegionName := "dst"
  let DK := p.dk
  let DV := p.dv
  let CK := p.ck
  let CV := p.cv
  let BK := p.bk
  let BV := p.bv
  let POLICY : Nat := 8
  with_triton_signature quant8Call in triton {
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
  }

def commuted8Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : RegionName := "src"
  let DST : RegionName := "dst"
  let DK := p.dk
  let DV := p.dv
  let CK := p.ck
  let CV := p.cv
  let BK := p.bk
  let BV := p.bv
  let POLICY : Nat := 8
  with_triton_signature quant8Call in triton {
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
  }

def kernel8 (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commuted8Kernel p else original8Kernel p

def quant4Call : CallSignature where
  pointers := [("K",.fp32),("V",.fp32),("KC",.u8),("VC",.u8),
    ("KSZ",.fp32),("VSZ",.fp32),("SRC",.i64),("DST",.i64)]
  scalars := []
  constexpr := ["DK","DV","CK","CV","BK","BV","POLICY"]
  functions := [quantRow]

def original4Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : RegionName := "src"
  let DST : RegionName := "dst"
  let DK := p.dk
  let DV := p.dv
  let CK := p.ck
  let CV := p.cv
  let BK := p.bk
  let BV := p.bv
  let POLICY : Nat := 4
  with_triton_signature quant4Call in triton {
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
  }

def commuted4Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : RegionName := "src"
  let DST : RegionName := "dst"
  let DK := p.dk
  let DV := p.dv
  let CK := p.ck
  let CV := p.cv
  let BK := p.bk
  let BV := p.bv
  let POLICY : Nat := 4
  with_triton_signature quant4Call in triton {
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
  }

def kernel4 (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commuted4Kernel p else original4Kernel p

end QuantSource

end VeriTile.Bench.Repairs.KvCacheFilling

namespace VeriTile.Bench.Repairs.KvCacheFilling.Shared
open VeriTile.Triton

def parameters (r : Request) : CopySource.Parameters :=
  ⟨r.dk, r.dv, r.ck, r.cv, r.ck.nextPowerOfTwo, r.cv.nextPowerOfTwo⟩

def kernel (version : Bool) (r : Request) : ComputeKernel :=
  match r.policy with
  | .copy => CopySource.kernel version (parameters r)
  | .int4 => QuantSource.kernel4 version (parameters r)
  | .int8 => QuantSource.kernel8 version (parameters r)

def original := kernel Bool.false
def candidate := kernel Bool.true

end VeriTile.Bench.Repairs.KvCacheFilling.Shared
