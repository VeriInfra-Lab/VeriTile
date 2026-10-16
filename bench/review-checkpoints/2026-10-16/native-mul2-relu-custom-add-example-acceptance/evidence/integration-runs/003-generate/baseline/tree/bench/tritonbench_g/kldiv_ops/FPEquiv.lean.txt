import bench.tritonbench_g.kldiv_ops.Proofs.FP
import bench.tritonbench_g.kldiv_ops.Proofs.Precision
import bench.tritonbench_g.kldiv_ops.Proofs.NumericExecution

namespace VeriTile.Bench.TritonBenchG.KldivOps.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forward_default_program (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : Program where
  kernel := forwardDefault pred target output v Y T O sy st so N B eps reduction
  domain := []

/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
specification forward_default_equivalence (pred target output : Storage)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (_hSource : ForwardDefaultSupported pred target output reduction) :
    forward_default_program pred target output Bool.false Y T O sy st so N B eps reduction ≡[assumptions]
      forward_default_program pred target output Bool.true Y T O sy st so N B eps reduction := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases pred <;> cases target <;> cases output <;> rfl)
  intro α _ M D _ s _
  exact FPProof.forward_default_equivalence M pred target output Y T O sy st so N B eps reduction s

#print_fp_assumptions forward_default_equivalence

def forward_log_program (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : Program where
  kernel := forwardLog pred target output v Y T O sy st so N B eps reduction
  domain := []

/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
specification forward_log_equivalence (pred : Storage) (target : MathStorage) (output : Storage)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (_hSource : ForwardLogSupported pred target output reduction) :
    forward_log_program pred target output Bool.false Y T O sy st so N B eps reduction ≡[assumptions]
      forward_log_program pred target output Bool.true Y T O sy st so N B eps reduction := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases pred <;> cases target <;> cases output <;> rfl)
  intro α _ M D _ s _
  exact FPProof.forward_log_equivalence M pred target output Y T O sy st so N B eps reduction s

#print_fp_assumptions forward_log_equivalence

def backward_default_program (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : Program where
  kernel := backwardDefault target output v T O st so N B
  domain := []

/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
specification backward_default_equivalence (target output : Storage)
    (T O : RegionName) (st so N B : Nat) (_hSource : BackwardSourceSupported target output Bool.false) :
    backward_default_program target output Bool.false T O st so N B ≡[assumptions]
      backward_default_program target output Bool.true T O st so N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases target <;> cases output <;> rfl)
  intro α _ M D _ s _
  exact FPProof.backward_default_equivalence M target output T O st so N B s

#print_fp_assumptions backward_default_equivalence

def backward_log_program (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : Program where
  kernel := backwardLog target output v T O st so N B
  domain := []

/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
specification backward_log_equivalence (target : MathStorage) (output : Storage)
    (T O : RegionName) (st so N B : Nat) (_hSource : BackwardSourceSupported target.storage output Bool.true) :
    backward_log_program target output Bool.false T O st so N B ≡[assumptions]
      backward_log_program target output Bool.true T O st so N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases target <;> cases output <;> rfl)
  intro α _ M D _ s _
  exact FPProof.backward_log_equivalence M target output T O st so N B s

#print_fp_assumptions backward_log_equivalence

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.FPEquiv
