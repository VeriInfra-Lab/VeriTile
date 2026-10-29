import bench.tritonbench_g.kldiv_ops.Proofs.ForwardPrecision

namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- The initializer remains the unannotated source zero (default fp32 channel),
even for formal profiles whose reduction
carry is rejected by the pinned frontend; SourceDomain keeps that distinction. -/
theorem forward_default_initializer (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardDefault pred target output v Y T O sy st so N B eps reduction).surfaceBody[5]? =
      some (.assign .real [] "loss_sum" (.alg (.const 0.0))) := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

/-- The full signed range and post-loop statement remain in both variants. -/
theorem forward_default_control (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    let source := forwardDefault pred target output v Y T O sy st so N B eps reduction
    source.surfaceBody.length = 8 ∧
      source.surfaceBody[6]? = some (.forRangeInt "i" (.constInt 0) (.constInt N)
        (.constNat B) (forwardBody source)) ∧
      (forwardBody source).length = 6 ∧ source.inputs = [Y,T] ∧ source.outputs = [O] := by
  cases pred <;> cases target <;> cases output <;> cases v <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- The initializer remains the unannotated source zero (default fp32 channel),
even for formal profiles whose reduction
carry is rejected by the pinned frontend; SourceDomain keeps that distinction. -/
theorem forward_log_initializer (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardLog pred target output v Y T O sy st so N B eps reduction).surfaceBody[5]? =
      some (.assign .real [] "loss_sum" (.alg (.const 0.0))) := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

/-- The full signed range and post-loop statement remain in both variants. -/
theorem forward_log_control (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    let source := forwardLog pred target output v Y T O sy st so N B eps reduction
    source.surfaceBody.length = 8 ∧
      source.surfaceBody[6]? = some (.forRangeInt "i" (.constInt 0) (.constInt N)
        (.constNat B) (forwardBody source)) ∧
      (forwardBody source).length = 6 ∧ source.inputs = [Y,T] ∧ source.outputs = [O] := by
  cases pred <;> cases target <;> cases output <;> cases v <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem backward_log_control (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    let source := backwardLog target output v T O st so N B
    source.surfaceBody.length = 6 ∧
      source.surfaceBody[5]? = some (.forRangeInt "i" (.constInt 0) (.constInt N)
        (.constNat B) (backwardBody source)) ∧
      (backwardBody source).length = 5 ∧ source.inputs = [T] ∧ source.outputs = [O] := by
  cases target <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- Every frontend-admitted nonzero reduction keeps the fp32 carry; the formal
fp64 branch checked by ForwardPrecision is not silently counted as admitted. -/
theorem source_supported_reduction_fp32 (pred target output : Storage)
    (flag : Bool) (reduction : Nat)
    (hSource : ForwardSourceSupported pred target output flag reduction)
    (hR : reduction ≠ 0) : ForwardPrecision.isWide pred target = Bool.false := by
  have hcarry : pred ≠ .fp64 ∧ target ≠ .fp64 := by
    rcases hSource.2 with he | h
    · exact False.elim (hR he)
    · exact h
  rcases hcarry with ⟨hp, ht⟩
  cases pred <;> cases target <;> simp_all [ForwardPrecision.isWide]

/-- Actual admitted default reduction has the independently specified fp32
block sum and fp32 carry update, including the unchanged elementwise arm. -/
theorem forward_default_reduction_precision (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (hSource : ForwardDefaultSupported pred target output reduction) (hR : reduction ≠ 0) :
    (forwardBody (forwardDefault pred target output v Y T O sy st so N B eps reduction))[5]? =
      some (ForwardPrecision.expectedBranch Bool.false output B reduction) := by
  have hw := source_supported_reduction_fp32 pred target output Bool.false reduction hSource hR
  simpa only [hw] using
    (ForwardPrecision.default_profiles pred target output v Y T O sy st so N B eps reduction).2.2.2.1

/-- The log-target branch has the same fp32 sum/carry discipline whenever the
unchanged source is admitted for nonzero reduction. -/
theorem forward_log_reduction_precision (pred : Storage) (target : MathStorage)
    (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (hSource : ForwardLogSupported pred target output reduction) (hR : reduction ≠ 0) :
    (forwardBody (forwardLog pred target output v Y T O sy st so N B eps reduction))[5]? =
      some (ForwardPrecision.expectedBranch Bool.false output B reduction) := by
  have hw := source_supported_reduction_fp32 pred target.storage output Bool.true reduction hSource hR
  simpa only [hw] using
    (ForwardPrecision.log_profiles pred target output v Y T O sy st so N B eps reduction).2.2.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
