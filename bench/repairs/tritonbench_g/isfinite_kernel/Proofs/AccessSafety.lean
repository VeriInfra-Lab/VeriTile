import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Source
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Every actual active load/store is backed by its declared strided window. -/
theorem access_safe {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B : Nat) (s : State A) (bounds : RegionName → Nat)
    (hx : ∀ j : Fin B,s.pids 0*B+j.val<n → input.address (s.pids 0*B+j.val)<bounds "x")
    (hy : ∀ j : Fin B,s.pids 0*B+j.val<n → output.address (s.pids 0*B+j.val)<bounds "y") :
    TraceSafe (algebra precision classify) bounds (source precision input output n B) s := by
  cases precision
  all_goals
    simp [TraceSafe,Safety.traceSafeList,Safety.stepSafe,Safety.safeAlg,Safety.safeExpr,
      Safety.safeCompute,Safety.safeOp_unfold,Safety.safeMask,Safety.safeAddress,Safety.addressSafe,
      Safety.maskActive,ComputeKernel.surfaceBody,source,source32,source64,View.address,View.signedAddress,
      FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      algebra,Precision.finite,ComputeDType.eraseDType,numeric,natLt,bop,FP.Structural.store,Region.cast,
      ofFloat,toFloat,Option.bind]
  all_goals exact ⟨fun j hj => hx j hj,fun j hj => hy j hj⟩

/-- Every actual active load/store is backed by its declared strided window. -/
theorem candidate_access_safe {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B : Nat) (s : State A) (bounds : RegionName → Nat)
    (hx : ∀ j : Fin B,s.pids 0*B+j.val<n → input.address (s.pids 0*B+j.val)<bounds "x")
    (hy : ∀ j : Fin B,s.pids 0*B+j.val<n → output.address (s.pids 0*B+j.val)<bounds "y") :
    TraceSafe (algebra precision classify) bounds (indexCommuted precision input output n B) s := by
  have hx' : ∀ j : Fin B,B*s.pids 0+j.val<n → input.address (B*s.pids 0+j.val)<bounds "x" := by
    simpa only [Nat.mul_comm] using hx
  have hy' : ∀ j : Fin B,B*s.pids 0+j.val<n → output.address (B*s.pids 0+j.val)<bounds "y" := by
    simpa only [Nat.mul_comm] using hy
  cases precision
  all_goals
    simp [TraceSafe,Safety.traceSafeList,Safety.stepSafe,Safety.safeAlg,Safety.safeExpr,
      Safety.safeCompute,Safety.safeOp_unfold,Safety.safeMask,Safety.safeAddress,Safety.addressSafe,
      Safety.maskActive,ComputeKernel.surfaceBody,indexCommuted,candidate32,candidate64,Nat.mul_comm,View.address,View.signedAddress,
      FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      algebra,Precision.finite,ComputeDType.eraseDType,numeric,natLt,bop,FP.Structural.store,Region.cast,
      ofFloat,toFloat,Option.bind]
  all_goals exact ⟨fun j hj => hx' j hj,fun j hj => hy' j hj⟩

end
end VeriTile.Bench.Repairs.IsfiniteKernel.Source
