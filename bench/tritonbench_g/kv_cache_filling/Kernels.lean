import bench.tritonbench_g.kv_cache_filling.Model
import VeriTile.Triton.DSL

/-! Complete copy, INT8 and INT4 source programs. The candidate commutes only
the integer product in the first key load. All other operations, precisions,
masks, casts and stores are identical. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling.CopySource
open VeriTile.Triton

structure Parameters where
  (dk dv ck cv bk bv : Nat)

def originalKernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let SRC : Region .int := ⟨"src"⟩
  let DST : Region .int := ⟨"dst"⟩
  triton {
    tl.region K = tl.float32, V = tl.float32,
      KC = tl.float32, VC = tl.float32, SRC = tl.int64, DST = tl.int64
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, $(p.bk)).to(tl.int64)
    iv = tl.arange(0, $(p.bv)).to(tl.int64)
    safe_ik = tl.where(ik < $(p.ck), ik, $((0 : Int)))
    safe_iv = tl.where(iv < $(p.cv), iv, $((0 : Int)))
    k = tl.load(K + src * $(p.dk) + safe_ik, mask=ik < $(p.dk), other=0.0)
    v = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.dv), other=0.0)
    tl.store(KC + dst * $(p.dk) + safe_ik, k, mask=ik < $(p.dk))
    tl.store(VC + dst * $(p.dv) + safe_iv, v, mask=iv < $(p.dv))
  }

def commutedKernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let SRC : Region .int := ⟨"src"⟩
  let DST : Region .int := ⟨"dst"⟩
  triton {
    tl.region K = tl.float32, V = tl.float32,
      KC = tl.float32, VC = tl.float32, SRC = tl.int64, DST = tl.int64
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, $(p.bk)).to(tl.int64)
    iv = tl.arange(0, $(p.bv)).to(tl.int64)
    safe_ik = tl.where(ik < $(p.ck), ik, $((0 : Int)))
    safe_iv = tl.where(iv < $(p.cv), iv, $((0 : Int)))
    k = tl.load(K + $(p.dk) * src + safe_ik, mask=ik < $(p.dk), other=0.0)
    v = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.dv), other=0.0)
    tl.store(KC + dst * $(p.dk) + safe_ik, k, mask=ik < $(p.dk))
    tl.store(VC + dst * $(p.dv) + safe_iv, v, mask=iv < $(p.dv))
  }

def kernel (version : Bool) : Parameters → ComputeKernel
  | p => if version then commutedKernel p else originalKernel p

end VeriTile.Bench.Repairs.KvCacheFilling.CopySource

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantSource
open VeriTile.Triton
abbrev Parameters := CopySource.Parameters

def original8Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : Region .int := ⟨"src"⟩
  let DST : Region .int := ⟨"dst"⟩
  triton {
    tl.region K = tl.float32, V = tl.float32, KC = tl.uint8, VC = tl.uint8,
      KSZ = tl.float32, VSZ = tl.float32, SRC = tl.int64, DST = tl.int64
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, $(p.bk)).to(tl.int64)
    iv = tl.arange(0, $(p.bv)).to(tl.int64)
    safe_ik = tl.where(ik < $(p.ck), ik, $((0 : Int)))
    safe_iv = tl.where(iv < $(p.cv), iv, $((0 : Int)))
    k0 = tl.load(K + src * $(p.dk))
    v0 = tl.load(V + src * $(p.dv))
    k = tl.load(K + src * $(p.dk) + safe_ik, mask=ik < $(p.dk), other=0.0)
    v = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.dv), other=0.0)
    k = tl.where(ik < $(p.dk), k, k0)
    v = tl.where(iv < $(p.dv), v, v0)
    klo = tl.min(k, 0)
    khi = tl.max(k, 0)
    kraw_scale = (khi - klo) / 255
    sk = tl.where(khi == klo, 1.0, kraw_scale)
    zk = -klo / sk
    qk = (k / sk + zk + 0.5).to(tl.uint8)
    vlo = tl.min(v, 0)
    vhi = tl.max(v, 0)
    vraw_scale = (vhi - vlo) / 255
    sv = tl.where(vhi == vlo, 1.0, vraw_scale)
    zv = -vlo / sv
    qv = (v / sv + zv + 0.5).to(tl.uint8)
    tl.store(KC + dst * $(p.ck) + safe_ik, qk, mask=ik < $(p.ck))
    tl.store(VC + dst * $(p.cv) + safe_iv, qv, mask=iv < $(p.cv))
    tl.store(KSZ + dst * $(2), sk)
    tl.store(KSZ + dst * $(2) + $(1), zk)
    tl.store(VSZ + dst * $(2), sv)
    tl.store(VSZ + dst * $(2) + $(1), zv)
  }

def commuted8Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : Region .int := ⟨"src"⟩
  let DST : Region .int := ⟨"dst"⟩
  triton {
    tl.region K = tl.float32, V = tl.float32, KC = tl.uint8, VC = tl.uint8,
      KSZ = tl.float32, VSZ = tl.float32, SRC = tl.int64, DST = tl.int64
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, $(p.bk)).to(tl.int64)
    iv = tl.arange(0, $(p.bv)).to(tl.int64)
    safe_ik = tl.where(ik < $(p.ck), ik, $((0 : Int)))
    safe_iv = tl.where(iv < $(p.cv), iv, $((0 : Int)))
    k0 = tl.load(K + src * $(p.dk))
    v0 = tl.load(V + src * $(p.dv))
    k = tl.load(K + $(p.dk) * src + safe_ik, mask=ik < $(p.dk), other=0.0)
    v = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.dv), other=0.0)
    k = tl.where(ik < $(p.dk), k, k0)
    v = tl.where(iv < $(p.dv), v, v0)
    klo = tl.min(k, 0)
    khi = tl.max(k, 0)
    kraw_scale = (khi - klo) / 255
    sk = tl.where(khi == klo, 1.0, kraw_scale)
    zk = -klo / sk
    qk = (k / sk + zk + 0.5).to(tl.uint8)
    vlo = tl.min(v, 0)
    vhi = tl.max(v, 0)
    vraw_scale = (vhi - vlo) / 255
    sv = tl.where(vhi == vlo, 1.0, vraw_scale)
    zv = -vlo / sv
    qv = (v / sv + zv + 0.5).to(tl.uint8)
    tl.store(KC + dst * $(p.ck) + safe_ik, qk, mask=ik < $(p.ck))
    tl.store(VC + dst * $(p.cv) + safe_iv, qv, mask=iv < $(p.cv))
    tl.store(KSZ + dst * $(2), sk)
    tl.store(KSZ + dst * $(2) + $(1), zk)
    tl.store(VSZ + dst * $(2), sv)
    tl.store(VSZ + dst * $(2) + $(1), zv)
  }

def kernel8 (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commuted8Kernel p else original8Kernel p

def original4Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : Region .int := ⟨"src"⟩
  let DST : Region .int := ⟨"dst"⟩
  triton {
    tl.region K = tl.float32, V = tl.float32, KC = tl.uint8, VC = tl.uint8,
      KSZ = tl.float32, VSZ = tl.float32, SRC = tl.int64, DST = tl.int64
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, $(p.bk)).to(tl.int64)
    iv = tl.arange(0, $(p.bv)).to(tl.int64)
    safe_ik = tl.where(ik < $(p.ck), ik, $((0 : Int)))
    safe_iv = tl.where(iv < $(p.cv), iv, $((0 : Int)))
    k10 = tl.load(K + src * $(p.dk))
    k20 = tl.load(K + src * $(p.dk) + $(p.ck))
    v10 = tl.load(V + src * $(p.dv))
    v20 = tl.load(V + src * $(p.dv) + $(p.cv))
    k1 = tl.load(K + src * $(p.dk) + safe_ik, mask=ik < $(p.ck), other=0.0).to(tl.float32)
    k2 = tl.load(K + src * $(p.dk) + $(p.ck) + safe_ik, mask=ik < $(p.ck), other=0.0).to(tl.float32)
    v1 = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.cv), other=0.0).to(tl.float32)
    v2 = tl.load(V + src * $(p.dv) + $(p.cv) + safe_iv, mask=iv < $(p.cv), other=0.0).to(tl.float32)
    k1 = tl.where(ik < $(p.ck), k1, k10)
    k2 = tl.where(ik < $(p.ck), k2, k20)
    v1 = tl.where(iv < $(p.cv), v1, v10)
    v2 = tl.where(iv < $(p.cv), v2, v20)
    klo = tl.min(tl.minimum(k1, k2), 0)
    khi = tl.max(tl.maximum(k1, k2), 0)
    vlo = tl.min(tl.minimum(v1, v2), 0)
    vhi = tl.max(tl.maximum(v1, v2), 0)
    kraw_scale = (khi - klo) / 15
    sk = tl.where(khi == klo, 1.0, kraw_scale)
    zk = -klo / sk
    qk1 = (k1 / sk + zk + 0.5).to(tl.uint8)
    kraw_scale2 = (khi - klo) / 15
    sk2 = tl.where(khi == klo, 1.0, kraw_scale2)
    zk2 = -klo / sk2
    qk2 = (k2 / sk2 + zk2 + 0.5).to(tl.uint8)
    vraw_scale = (vhi - vlo) / 15
    sv = tl.where(vhi == vlo, 1.0, vraw_scale)
    zv = -vlo / sv
    qv1 = (v1 / sv + zv + 0.5).to(tl.uint8)
    vraw_scale2 = (vhi - vlo) / 15
    sv2 = tl.where(vhi == vlo, 1.0, vraw_scale2)
    zv2 = -vlo / sv2
    qv2 = (v2 / sv2 + zv2 + 0.5).to(tl.uint8)
    qk = (qk1 + qk2 * $(16)).to(tl.uint8)
    qv = (qv1 + qv2 * $(16)).to(tl.uint8)
    tl.store(KC + dst * $(p.ck) + safe_ik, qk, mask=ik < $(p.ck))
    tl.store(VC + dst * $(p.cv) + safe_iv, qv, mask=iv < $(p.cv))
    tl.store(KSZ + dst * $(2), sk)
    tl.store(KSZ + dst * $(2) + $(1), zk)
    tl.store(VSZ + dst * $(2), sv)
    tl.store(VSZ + dst * $(2) + $(1), zv)
  }

def commuted4Kernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KC : RegionName := "keyCache"
  let VC : RegionName := "valueCache"
  let KSZ : RegionName := "keyMeta"
  let VSZ : RegionName := "valueMeta"
  let SRC : Region .int := ⟨"src"⟩
  let DST : Region .int := ⟨"dst"⟩
  triton {
    tl.region K = tl.float32, V = tl.float32, KC = tl.uint8, VC = tl.uint8,
      KSZ = tl.float32, VSZ = tl.float32, SRC = tl.int64, DST = tl.int64
    job = tl.program_id(0).to(tl.int64)
    src = tl.load(SRC + job)
    dst = tl.load(DST + job)
    ik = tl.arange(0, $(p.bk)).to(tl.int64)
    iv = tl.arange(0, $(p.bv)).to(tl.int64)
    safe_ik = tl.where(ik < $(p.ck), ik, $((0 : Int)))
    safe_iv = tl.where(iv < $(p.cv), iv, $((0 : Int)))
    k10 = tl.load(K + src * $(p.dk))
    k20 = tl.load(K + src * $(p.dk) + $(p.ck))
    v10 = tl.load(V + src * $(p.dv))
    v20 = tl.load(V + src * $(p.dv) + $(p.cv))
    k1 = tl.load(K + $(p.dk) * src + safe_ik, mask=ik < $(p.ck), other=0.0).to(tl.float32)
    k2 = tl.load(K + src * $(p.dk) + $(p.ck) + safe_ik, mask=ik < $(p.ck), other=0.0).to(tl.float32)
    v1 = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.cv), other=0.0).to(tl.float32)
    v2 = tl.load(V + src * $(p.dv) + $(p.cv) + safe_iv, mask=iv < $(p.cv), other=0.0).to(tl.float32)
    k1 = tl.where(ik < $(p.ck), k1, k10)
    k2 = tl.where(ik < $(p.ck), k2, k20)
    v1 = tl.where(iv < $(p.cv), v1, v10)
    v2 = tl.where(iv < $(p.cv), v2, v20)
    klo = tl.min(tl.minimum(k1, k2), 0)
    khi = tl.max(tl.maximum(k1, k2), 0)
    vlo = tl.min(tl.minimum(v1, v2), 0)
    vhi = tl.max(tl.maximum(v1, v2), 0)
    kraw_scale = (khi - klo) / 15
    sk = tl.where(khi == klo, 1.0, kraw_scale)
    zk = -klo / sk
    qk1 = (k1 / sk + zk + 0.5).to(tl.uint8)
    kraw_scale2 = (khi - klo) / 15
    sk2 = tl.where(khi == klo, 1.0, kraw_scale2)
    zk2 = -klo / sk2
    qk2 = (k2 / sk2 + zk2 + 0.5).to(tl.uint8)
    vraw_scale = (vhi - vlo) / 15
    sv = tl.where(vhi == vlo, 1.0, vraw_scale)
    zv = -vlo / sv
    qv1 = (v1 / sv + zv + 0.5).to(tl.uint8)
    vraw_scale2 = (vhi - vlo) / 15
    sv2 = tl.where(vhi == vlo, 1.0, vraw_scale2)
    zv2 = -vlo / sv2
    qv2 = (v2 / sv2 + zv2 + 0.5).to(tl.uint8)
    qk = (qk1 + qk2 * $(16)).to(tl.uint8)
    qv = (qv1 + qv2 * $(16)).to(tl.uint8)
    tl.store(KC + dst * $(p.ck) + safe_ik, qk, mask=ik < $(p.ck))
    tl.store(VC + dst * $(p.cv) + safe_iv, qv, mask=iv < $(p.cv))
    tl.store(KSZ + dst * $(2), sk)
    tl.store(KSZ + dst * $(2) + $(1), zk)
    tl.store(VSZ + dst * $(2), sv)
    tl.store(VSZ + dst * $(2) + $(1), zv)
  }

def kernel4 (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commuted4Kernel p else original4Kernel p

end VeriTile.Bench.Repairs.KvCacheFilling.QuantSource

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
