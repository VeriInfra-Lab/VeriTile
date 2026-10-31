import bench.tritonbench_g.layer_norm_fwd.Proofs.Pair
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Pair
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 6000

/-- The changed two-statement prefix only forms an integer pointer; neither
order of the row/stride multiplication accesses memory. -/
theorem candidate_prefix_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList A bounds
      ((indexCommuted f r stride M N eps p).kernel.surfaceBody.take 2) s := by
  simp [indexCommuted, Source.indexCommuted, ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeOp_unfold, step, evalExpr, evalOp_unfold]

/-- Safety is transported through the pure prefix's equal actual successor,
then through the literally identical complete suffix. Execution equality alone
is not used as a safety certificate. -/
theorem trace_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (bounds : RegionName → Nat) (s : State α)
    (safe : TraceSafe A bounds (original f r stride M N eps p).kernel s) :
    TraceSafe A bounds (indexCommuted f r stride M N eps p).kernel s := by
  change Safety.traceSafeList A bounds _ s at safe ⊢
  rw [← List.take_append_drop 2 (original f r stride M N eps p).kernel.surfaceBody,
    Safety.traceSafeList_append] at safe
  rw [← List.take_append_drop 2 (indexCommuted f r stride M N eps p).kernel.surfaceBody,
    Safety.traceSafeList_append]
  refine ⟨candidate_prefix_safe A f r stride M N eps p bounds s, ?_⟩
  intro t executed
  have hprefix := FPProof.prefix_run f r stride M N eps p (tile N) (tile_arange N) A s
  have hsuffix := FPProof.suffix f r stride M N eps p (tile N) (tile_arange N)
  exact hsuffix ▸ safe.2 t (hprefix.trans executed)

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Pair
