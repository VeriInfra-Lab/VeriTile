import bench.tritonbench_g.kv_cache_filling.Proofs.SharedReadback
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.KvCacheFilling.IOSafety
open VeriTile.Triton FP.Structural
open QuantProfile CopySource CopyExecution QuantExecution QuantValues
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem quant8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv) (pk : 0 < p.ck) (pv : 0 < p.cv)
    (shapeK : p.dk = p.ck) (shapeV : p.dv = p.cv) (version : Bool)
    (bounds : RegionName → Nat) (bs : s.pids 0 < bounds "src") (bd : s.pids 0 < bounds "dst")
    (bk : ∀ i, i < p.dk → src*p.dk+i < bounds "key")
    (bv : ∀ i, i < p.dv → src*p.dv+i < bounds "value")
    (bkc : ∀ i, i < p.ck → dst*p.ck+i < bounds "keyCache")
    (bvc : ∀ i, i < p.cv → dst*p.cv+i < bounds "valueCache")
    (bkm : ∀ i, i < 2 → dst*2+i < bounds "keyMeta")
    (bvm : ∀ i, i < 2 → dst*2+i < bounds "valueMeta") :
    FP.Structural.TraceSafe (algebra a) bounds (QuantSource.kernel8 version p) s := by
  have bk0 : src*p.ck < bounds "key" := by
    simpa only [shapeK, Nat.add_zero] using bk 0 (by omega)
  have bv0 : src*p.cv < bounds "value" := by
    simpa only [shapeV, Nat.add_zero] using bv 0 (by omega)
  have bkm0 : dst*2 < bounds "keyMeta" := by simpa only [Nat.add_zero] using bkm 0 (by omega)
  have bvm0 : dst*2 < bounds "valueMeta" := by simpa only [Nat.add_zero] using bvm 0 (by omega)
  cases version
  all_goals
    simp (maxSteps := 2000000) [FP.Structural.TraceSafe, QuantSource.kernel8, QuantSource.original8Kernel,original8Kernel, QuantSource.commuted8Kernel,candidate8Kernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, store, numeric, numericLt, numericEq, bop,
      ComputeDType.eraseDType, Region.cast, Option.bind, hs, hd, algebra,
      ofFloat, toFloat, TileShape.axisDim, State.setReg, hk, hv, Int.mul_comm,
      index_toNat, metadata_index_toNat, mul_index_toNat, add_index_toNat,
      meta0_index_toNat, meta1_index_toNat, shapeK, shapeV, bs, bd]
  all_goals simp only [shapeK, shapeV] at bk bv
  all_goals aesop

#axiomsClean quant8
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.IOSafety
