import bench.tritonbench_g.kldiv_ops.Proofs.SourceExecution
import bench.tritonbench_g.kldiv_ops.Proofs.ForwardSafety
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardSafety

namespace VeriTile.Bench.TritonBenchG.KldivOps.Source
open VeriTile.Triton TypedSource

/-- The candidate prefix accesses no memory and leaves precisely the same state. -/
theorem forward_math_safe (bounds : RegionBounds) (s : BlockState) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (flag : Bool) (reduction : Nat)
    (hs : (FullAlgorithm.forward Y T O sy st so N B eps flag reduction).TraceSafeR .triv bounds s) :
    (forwardMath v Y T O sy st so N B eps flag reduction).TraceSafeR .triv bounds s := by
  rw [← forward_math_original Y T O sy st so N B eps flag reduction] at hs
  change Stmt.TraceSafeListR .triv bounds
    (forwardPrefix v Y T O sy st so ++
      (FullAlgorithm.forward Y T O sy st so N B eps flag reduction).body.drop 4) s
  change Stmt.TraceSafeListR .triv bounds
    (forwardPrefix Bool.false Y T O sy st so ++
      (FullAlgorithm.forward Y T O sy st so N B eps flag reduction).body.drop 4) s at hs
  rw [safe_append_iff] at hs ⊢
  refine ⟨forward_prefix_safe bounds v Y T O sy st so s, ?_⟩
  intro t he
  apply hs.2 t
  cases v
  · exact he
  · exact (forward_prefix_exec Y T O sy st so s).symm.trans he

theorem backward_math_safe (bounds : RegionBounds) (s : BlockState) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (flag : Bool)
    (hs : (FullAlgorithm.backward T O st so N B flag).TraceSafeR .triv bounds s) :
    (backwardMath v T O st so N B flag).TraceSafeR .triv bounds s := by
  rw [← backward_math_original T O st so N B flag] at hs
  change Stmt.TraceSafeListR .triv bounds
    (backwardPrefix v T O st so ++ (FullAlgorithm.backward T O st so N B flag).body.drop 3) s
  change Stmt.TraceSafeListR .triv bounds
    (backwardPrefix Bool.false T O st so ++
      (FullAlgorithm.backward T O st so N B flag).body.drop 3) s at hs
  rw [safe_append_iff] at hs ⊢
  refine ⟨backward_prefix_safe bounds v T O st so s, ?_⟩
  intro t he
  apply hs.2 t
  cases v
  · exact he
  · exact (backward_prefix_exec T O st so s).symm.trans he

/-- All source reads and writes are bounded; reduction permits aliases because
its scalar output store follows all reads. -/
theorem forward_algorithm_safe (bounds : RegionBounds) (s : BlockState)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (flag : Bool) (reduction : Nat)
    (hB : 0 < B) (hSep : reduction = 0 → Y ≠ O ∧ T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : reduction = 0 → ∀ k : Fin N, s.pids 0 * so + k.val < bounds O)
    (hscalar : reduction ≠ 0 → s.pids 0 * so < bounds O) :
    (FullAlgorithm.forward Y T O sy st so N B eps flag reduction).TraceSafeR .triv bounds s := by
  by_cases hr : reduction = 0
  · subst reduction
    exact FullForward.element_source_safe bounds s Y T O sy st so N B eps flag
      (hSep rfl).1 (hSep rfl).2 hby hbt (hbo rfl)
  · exact FullForward.reduction_source_safe bounds s Y T O sy st so N B eps flag
      reduction hB hr hby hbt (hscalar hr)

theorem forward_default_safe (bounds : RegionBounds) (s : BlockState)
    (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (hB : 0 < B) (hSep : reduction = 0 → Y ≠ O ∧ T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : reduction = 0 → ∀ k : Fin N, s.pids 0 * so + k.val < bounds O)
    (hscalar : reduction ≠ 0 → s.pids 0 * so < bounds O) :
    (forwardDefault pred target output v Y T O sy st so N B eps reduction).eraseDType.toAlgKernel.TraceSafeR
      .triv bounds s := by
  rw [forward_default_projection]
  exact forward_math_safe bounds s v Y T O sy st so N B eps Bool.false reduction
    (forward_algorithm_safe bounds s Y T O sy st so N B eps Bool.false reduction
      hB hSep hby hbt hbo hscalar)

theorem forward_log_safe (bounds : RegionBounds) (s : BlockState)
    (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (hB : 0 < B) (hSep : reduction = 0 → Y ≠ O ∧ T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : reduction = 0 → ∀ k : Fin N, s.pids 0 * so + k.val < bounds O)
    (hscalar : reduction ≠ 0 → s.pids 0 * so < bounds O) :
    (forwardLog pred target output v Y T O sy st so N B eps reduction).eraseDType.toAlgKernel.TraceSafeR
      .triv bounds s := by
  rw [forward_log_projection]
  exact forward_math_safe bounds s v Y T O sy st so N B eps Bool.true reduction
    (forward_algorithm_safe bounds s Y T O sy st so N B eps Bool.true reduction
      hB hSep hby hbt hbo hscalar)

theorem backward_default_safe (bounds : RegionBounds) (s : BlockState)
    (target output : Storage) (v : Bool) (T O : RegionName) (st so N B : Nat)
    (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < bounds O) :
    (backwardDefault target output v T O st so N B).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  rw [backward_default_projection]
  exact backward_math_safe bounds s v T O st so N B Bool.false
    (FullBackward.source_safe bounds s T O st so N B Bool.false hTO hbt hbo)

theorem backward_log_safe (bounds : RegionBounds) (s : BlockState)
    (target : MathStorage) (output : Storage) (v : Bool) (T O : RegionName) (st so N B : Nat)
    (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < bounds O) :
    (backwardLog target output v T O st so N B).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  rw [backward_log_projection]
  exact backward_math_safe bounds s v T O st so N B Bool.true
    (FullBackward.source_safe bounds s T O st so N B Bool.true hTO hbt hbo)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.Source
