import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.RawPair
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RawPair
open VeriTile.Triton FP.Structural
open Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

/-- The actual changed prefix forms an integer pointer and performs no load or store. -/
theorem candidate_prefix_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (block : Nat) (hb : Random.ArangeSizeDomain block)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList A bounds
      ((indexCommuted f r stride M N eps p block hb).surfaceBody.take 2) s := by
  by_cases hN : N = 1
  all_goals simp [indexCommuted,rawIndexCommuted,rawIndexCommutedOne,hN,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,step,evalExpr,evalOp_unfold]

/-- Full safety follows through the proved pure prefix and the identical suffix;
whole-state execution equality alone is not treated as an access-safety proof. -/
theorem trace_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (block : Nat) (hb : Random.ArangeSizeDomain block)
    (bounds : RegionName → Nat) (s : State α)
    (safe : TraceSafe A bounds (kernel f r stride M N eps p block hb) s) :
    TraceSafe A bounds (indexCommuted f r stride M N eps p block hb) s := by
  change Safety.traceSafeList A bounds _ s at safe ⊢
  rw [← List.take_append_drop 2 (kernel f r stride M N eps p block hb).surfaceBody,
    Safety.traceSafeList_append] at safe
  rw [← List.take_append_drop 2 (indexCommuted f r stride M N eps p block hb).surfaceBody,
    Safety.traceSafeList_append]
  refine ⟨candidate_prefix_safe A f r stride M N eps p block hb bounds s,?_⟩
  intro t executed
  have hp := prefix_run f r stride M N eps p block hb A s
  have hs := suffix f r stride M N eps p block hb
  exact hs ▸ safe.2 t (hp.trans executed)

#axiomsClean candidate_prefix_safe
#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RawPair
