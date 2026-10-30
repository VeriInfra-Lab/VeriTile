import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SourceBlocks
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling.LegacyCopy
open Lean VeriTile.Triton

/-! Exact policy-0 specialization of `_fill_rows`. The quantized specializations
still require shared minimum-reduction support; this module is reusable partial
progress and does not claim acceptance of the entire three-policy task.
The candidate only commutes the integer multiplication forming K's source-row
offset. No floating operation, guard, Python implementation, or perf claim is
added. `ck = dk`, `cv = dv` are the existing policy-0 shape equations. -/

abbrev Parameters := CopySource.Parameters

syntax "kvFillingCopyLegacy%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(kvFillingCopyLegacy%[$version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let keyLoad : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| k = tl.load(K + src * $(p.dk) + safe_ik, mask=ik < $(p.dk), other=0.0))
      else
      `(tritonStmt| k = tl.load(K + $(p.dk) * src + safe_ik, mask=ik < $(p.dk), other=0.0))
    `(fun (p : Parameters) =>
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
        $keyLoad
        v = tl.load(V + src * $(p.dv) + safe_iv, mask=iv < $(p.dv), other=0.0)
        tl.store(KC + dst * $(p.dk) + safe_ik, k, mask=ik < $(p.dk))
        tl.store(VC + dst * $(p.dv) + safe_iv, v, mask=iv < $(p.dv))
      })

def kernel (commuted : Bool) : Parameters → ComputeKernel :=
  match commuted with
  | Bool.false => kvFillingCopyLegacy%[original]
  | Bool.true => kvFillingCopyLegacy%[commuted]

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KvCacheFilling.LegacyCopy


namespace VeriTile.Bench.Repairs.KvCacheFilling.LegacyQuant
open Lean VeriTile.Triton
abbrev Parameters := CopySource.Parameters

/-! Exact policy-8/4 source specializations. `_quant_row` is inlined in place.
The shared minimum/reduceMin callbacks preserve the complete ordered operation.
The source's implicit uint8 arithmetic narrowing is written explicitly at the
packed byte expression; `PtxCast.pack8` and `pack16_low8` justify this expansion.
Neither the Python program nor its assertions are changed. -/

syntax "kvFilling8Legacy%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(kvFilling8Legacy%[$version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let keyLoad : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| k = tl.load(K + src * $(p.dk) + safe_ik, mask=ik < $(p.dk), other=0.0))
      else
      `(tritonStmt| k = tl.load(K + $(p.dk) * src + safe_ik, mask=ik < $(p.dk), other=0.0))
    `(fun (p : Parameters) =>
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
        $keyLoad
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
      })

syntax "kvFilling4Legacy%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(kvFilling4Legacy%[$version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let keyLoad : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| k1 = tl.load(K + src * $(p.dk) + safe_ik, mask=ik < $(p.ck), other=0.0).to(tl.float32))
      else
      `(tritonStmt| k1 = tl.load(K + $(p.dk) * src + safe_ik, mask=ik < $(p.ck), other=0.0).to(tl.float32))
    `(fun (p : Parameters) =>
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
        $keyLoad
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
      })

def kernel8 (v : Bool) : Parameters → ComputeKernel :=
  match v with
  | Bool.false => kvFilling8Legacy%[original]
  | Bool.true => kvFilling8Legacy%[commuted]

def kernel4 (v : Bool) : Parameters → ComputeKernel :=
  match v with
  | Bool.false => kvFilling4Legacy%[original]
  | Bool.true => kvFilling4Legacy%[commuted]

end VeriTile.Bench.Repairs.KvCacheFilling.LegacyQuant

namespace VeriTile.Bench.Repairs.KvCacheFilling.SourcePreservation
open VeriTile.Triton FP.Structural

/-- Each current quotation contains the complete eight-statement Python body,
including all policies. The legacy blocks above remain historical specializations;
their ASTs omit the policy control and helper scope and are not source identities. -/
theorem copy_source (p : CopySource.Parameters) (version : Bool) :
    (CopySource.kernel version p).surfaceBody.length = 8 := by cases version <;> rfl

theorem int8_source (p : CopySource.Parameters) (version : Bool) :
    (QuantSource.kernel8 version p).surfaceBody.length = 8 := by cases version <;> rfl

theorem int4_source (p : CopySource.Parameters) (version : Bool) :
    (QuantSource.kernel4 version p).surfaceBody.length = 8 := by cases version <;> rfl

/-- The full raw source, including the caller policy test, runs exactly its
source-derived quantized branch for every model and initial state. -/
theorem int8_execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : CopySource.Parameters) (version : Bool) (s : State α) :
    exec M (QuantSource.kernel8 version p) s = run M (SourceBlocks.body8 version p) s :=
  SourceBlocks.execute8 M version p s

theorem int4_execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : CopySource.Parameters) (version : Bool) (s : State α) :
    exec M (QuantSource.kernel4 version p) s = run M (SourceBlocks.body4 version p) s :=
  SourceBlocks.execute4 M version p s

#axiomsClean copy_source
#axiomsClean int8_source
#axiomsClean int4_source
#axiomsClean int8_execution
#axiomsClean int4_execution
end VeriTile.Bench.Repairs.KvCacheFilling.SourcePreservation
