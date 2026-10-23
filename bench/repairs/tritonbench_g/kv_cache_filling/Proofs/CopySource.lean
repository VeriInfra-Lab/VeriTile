import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.KvCacheFilling.CopySource
open Lean VeriTile.Triton

/-! Exact policy-0 specialization of `_fill_rows`. The quantized specializations
still require shared minimum-reduction support; this module is reusable partial
progress and does not claim acceptance of the entire three-policy task.
The candidate only commutes the integer multiplication forming K's source-row
offset. No floating operation, guard, Python implementation, or perf claim is
added. `ck = dk`, `cv = dv` are the existing policy-0 shape equations. -/

structure Parameters where
  (dk dv ck cv bk bv : Nat)

syntax "kvFillingCopy%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(kvFillingCopy%[$version:ident]) => do
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
  | Bool.false => kvFillingCopy%[original]
  | Bool.true => kvFillingCopy%[commuted]

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KvCacheFilling.CopySource
