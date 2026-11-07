import bench.tritonbench_g.kv_cache_filling.Proofs.SharedReadback
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.KvCacheFilling.IOSafety
open VeriTile.Triton FP.Structural
open QuantProfile CopySource CopyExecution QuantExecution QuantValues
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem copy {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (shapeK : p.dk = p.ck) (shapeV : p.dv = p.cv) (version : Bool)
    (bounds : RegionName → Nat) (bs : s.pids 0 < bounds "src") (bd : s.pids 0 < bounds "dst")
    (bk : ∀ i, i < p.dk → src*p.dk+i < bounds "key")
    (bv : ∀ i, i < p.dv → src*p.dv+i < bounds "value")
    (bkc : ∀ i, i < p.ck → dst*p.ck+i < bounds "keyCache")
    (bvc : ∀ i, i < p.cv → dst*p.cv+i < bounds "valueCache") :
    FP.Structural.TraceSafe (algebra a) bounds (CopySource.kernel version p) s := by
  cases version
  all_goals
    simp (maxSteps := 1500000) [FP.Structural.TraceSafe, CopySource.kernel, CopySource.originalKernel,original0Kernel, CopySource.commutedKernel,candidate0Kernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, store, numeric, numericLt, bop,
      ComputeDType.eraseDType, Region.cast, Option.bind, hs, hd, algebra,
      ofFloat, toFloat, TileShape.axisDim, State.setReg, Int.mul_comm,
      index_toNat, shapeK, shapeV, bs, bd]
  all_goals simp only [shapeK, shapeV] at bk bv
  all_goals aesop

#axiomsClean copy
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.IOSafety
