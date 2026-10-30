import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.PairSource
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairSource
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math
open FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem traceSafe_of_steps [Inhabited α] (M N : Algebra α) (code : List ComputeStmt)
    (runs : ∀ st ∈ code, ∀ s, FP.Structural.step M st s = FP.Structural.step N st s)
    (safe : ∀ st ∈ code, ∀ bounds s, stepSafe M bounds st s ↔ stepSafe N bounds st s)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList M bounds code s ↔ traceSafeList N bounds code s := by
  induction code generalizing s with
  | nil => simp [traceSafeList_unfold]
  | cons st rest ih =>
    simp only [traceSafeList_cons, runs st (by simp), safe st (by simp)]
    constructor <;> rintro ⟨first, tail⟩ <;> refine ⟨first, ?_⟩
    · intro t ran
      exact (ih (fun q hq => runs q (by simp [hq]))
        (fun q hq => safe q (by simp [hq])) t).mp (tail t ran)
    · intro t ran
      exact (ih (fun q hq => runs q (by simp [hq]))
        (fun q hq => safe q (by simp [hq])) t).mpr (tail t ran)

theorem before_step_safe [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) : ∀ st ∈ before c hasBias useSoftplus, ∀ bounds s,
    stepSafe (engine (reverseMul M)) bounds st s ↔ stepSafe (engine M) bounds st s := by
  cases hasBias <;> cases useSoftplus
  all_goals
    simp [before, Source.prefixBefore, kernel, source, ComputeKernel.surfaceBody]
    repeat' apply And.intro
    all_goals intro bounds s
    all_goals simp [traceSafeList_unfold, stepSafe_unfold, safeOp_unfold, safeCompute_unfold,
      safeExpr_unfold, safeAddress_unfold, safeMask_unfold, addressSafe_unfold, maskActive_unfold,
      run, FP.Structural.step, evalExpr, evalComputeOp, evalOp_unfold,
      engine, Algebra.withDefaultPrecision, reverseMul, resolvePrecision, ComputeDType.eraseDType,
      numeric, numericLe, numericLt, natLe, natLt]

theorem after_step_safe [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) : ∀ st ∈ after c hasBias useSoftplus, ∀ bounds s,
    stepSafe (engine (reverseMul M)) bounds st s ↔ stepSafe (engine M) bounds st s := by
  cases hasBias <;> cases useSoftplus
  all_goals
    simp [after, before, Source.prefixBefore, kernel, source, ComputeKernel.surfaceBody]
    repeat' apply And.intro
    all_goals intro bounds s
    all_goals simp [traceSafeList_unfold, stepSafe_unfold, safeOp_unfold, safeCompute_unfold,
      safeExpr_unfold, safeAddress_unfold, safeMask_unfold, addressSafe_unfold, maskActive_unfold,
      run, FP.Structural.step, evalExpr, evalComputeOp, evalOp_unfold,
      engine, Algebra.withDefaultPrecision, reverseMul, resolvePrecision, ComputeDType.eraseDType,
      numeric, numericLe, numericLt, natLe, natLt]

theorem before_safe [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (reverseMul M)) bounds (before c hasBias useSoftplus) s ↔
      traceSafeList (engine M) bounds (before c hasBias useSoftplus) s := by
  exact traceSafe_of_steps _ _ _ (before_steps M c hasBias useSoftplus)
    (before_step_safe M c hasBias useSoftplus) bounds s

theorem after_safe [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (reverseMul M)) bounds (after c hasBias useSoftplus) s ↔
      traceSafeList (engine M) bounds (after c hasBias useSoftplus) s := by
  exact traceSafe_of_steps _ _ _ (after_steps M c hasBias useSoftplus)
    (after_step_safe M c hasBias useSoftplus) bounds s

theorem update_safe [Inhabited α] (M : Algebra α) (c : Geometry)
    (bounds : RegionName → Nat) (s : State α) (reverse : Bool) :
    traceSafeList (engine M) bounds [updateStmt c reverse] s := by
  apply traceSafeList_cons_intro
  · cases reverse <;>
      simp [updateStmt, stepSafe_unfold, safeExpr_unfold, safeCompute_unfold, safeOp_unfold]
  · intro t _
    simp [traceSafeList_unfold]

/-- Every source memory access and post-update continuation is retained. -/
theorem complete_safe [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe (engine (reverseMul M)) bounds (kernel c hasBias useSoftplus) s ↔
      TraceSafe (engine M) bounds (candidateKernel c hasBias useSoftplus) s := by
  have hb := funext (before_run M c hasBias useSoftplus)
  have hu := funext (update_run M c)
  simp only [TraceSafe, original_body, candidate_body, List.append_assoc,
    traceSafeList_append, before_safe, hb, hu, update_safe, after_safe]

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairSource
