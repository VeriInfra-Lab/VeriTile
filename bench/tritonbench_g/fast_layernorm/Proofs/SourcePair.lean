import bench.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.FastLayernorm.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Select the complete quoted source, including its actual caller bindings. -/
def source (backward candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :=
  if backward then
    if candidate then backwardIndexCommuted "dx" "dy" "x" "w" "r" "mu" N B
    else backwardKernel "dx" "dy" "x" "w" "r" "mu" N B
  else if candidate then forwardIndexCommuted "y" "x" "w" "b" "r" "mu" N B eps bias
  else forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias
private def position (backward : Bool) : Nat := if backward then 10 else 9
private def indexCode (backward candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :=
  ((source backward candidate N B eps bias).surfaceBody.drop (position backward)).take 1

private theorem index_run {A : Type} [Inhabited A] (M : Algebra A)
    (backward : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) (s : State A) :
    run M (indexCode backward Bool.false N B eps bias) s=
      run M (indexCode backward Bool.true N B eps bias) s := by
  cases backward <;> cases hr : s.regs .int [] "row_idx" <;>
    cases hs : s.regs .int [] "X_row_stride"
  all_goals simp (maxSteps := 1000000) [indexCode,position,source,
    forwardKernel,backwardKernel,forwardIndexCommuted,backwardIndexCommuted,
    forwardBodyPastedKernel,backwardBodyPastedKernel,forwardCandidateKernel,backwardCandidateKernel,
    ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.integer,ScalarArgument.fp32,
    ComputeKernel.surfaceBody,List.drop,List.take,run,step,evalExpr,evalOp_unfold,
    numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,hr,hs,Option.bind,Int.mul_comm]
  all_goals try (congr 1; funext i; simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm])

private theorem index_safe {A : Type} [Inhabited A] (M : Algebra A) (bounds : RegionName → Nat)
    (backward candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) (s : State A) :
    Safety.traceSafeList M bounds (indexCode backward candidate N B eps bias) s := by
  cases backward <;> cases candidate
  all_goals simp [indexCode,position,source,
    forwardKernel,backwardKernel,forwardIndexCommuted,backwardIndexCommuted,
    forwardBodyPastedKernel,backwardBodyPastedKernel,forwardCandidateKernel,backwardCandidateKernel,
    ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.integer,ScalarArgument.fp32,
    ComputeKernel.surfaceBody,List.drop,List.take,Safety.traceSafeList,Safety.stepSafe,
    Safety.safeExpr,Safety.safeAlg,Safety.safeOp,Safety.safeOp_unfold]
  all_goals split <;> trivial

private theorem split_source (backward candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :
    (source backward candidate N B eps bias).surfaceBody=
      (source backward candidate N B eps bias).surfaceBody.take (position backward) ++
        (indexCode backward candidate N B eps bias ++
          (source backward candidate N B eps bias).surfaceBody.drop (position backward+1)) := by
  cases backward <;> cases candidate <;> rfl
private theorem prefix_eq (backward : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :
    (source backward Bool.false N B eps bias).surfaceBody.take (position backward)=
      (source backward Bool.true N B eps bias).surfaceBody.take (position backward) := by
  cases backward <;> rfl
private theorem suffix_eq (backward : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :
    (source backward Bool.false N B eps bias).surfaceBody.drop (position backward+1)=
      (source backward Bool.true N B eps bias).surfaceBody.drop (position backward+1) := by
  cases backward <;> rfl

private theorem same_run {A : Type} [Inhabited A] (M : Algebra A) (backward : Bool)
    (N B : Nat) (eps : ℝ) (bias : Bool) (s : State A) :
    exec M (source backward Bool.false N B eps bias) s=
      exec M (source backward Bool.true N B eps bias) s := by
  change run M (source backward Bool.false N B eps bias).surfaceBody s=
    run M (source backward Bool.true N B eps bias).surfaceBody s
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [run_append,prefix_eq,index_run,suffix_eq]
private theorem same_safe {A : Type} [Inhabited A] (M : Algebra A) (bounds : RegionName → Nat)
    (backward : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) (s : State A) :
    TraceSafe M bounds (source backward Bool.false N B eps bias) s ↔
      TraceSafe M bounds (source backward Bool.true N B eps bias) s := by
  unfold TraceSafe
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [Safety.traceSafeList_append,prefix_eq,index_safe,index_run,suffix_eq,true_and]

def forward (candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) := source Bool.false candidate N B eps bias
def backward (candidate : Bool) (N B : Nat) := source Bool.true candidate N B 0 Bool.false

theorem forward_exec {A : Type} [Inhabited A] (M : Algebra A) (s : State A)
    (N B : Nat) (eps : ℝ) (bias : Bool) :
    exec M (forward Bool.false N B eps bias) s=exec M (forward Bool.true N B eps bias) s :=
  same_run M Bool.false N B eps bias s

theorem backward_exec {A : Type} [Inhabited A] (M : Algebra A) (s : State A) (N B : Nat) :
    exec M (backward Bool.false N B) s=exec M (backward Bool.true N B) s :=
  same_run M Bool.true N B 0 Bool.false s

theorem forward_safe {A : Type} [Inhabited A] (M : Algebra A) (s : State A)
    (bounds : RegionName → Nat) (N B : Nat) (eps : ℝ) (bias : Bool) :
    TraceSafe M bounds (forward Bool.false N B eps bias) s ↔
      TraceSafe M bounds (forward Bool.true N B eps bias) s := same_safe M bounds Bool.false N B eps bias s

theorem backward_safe {A : Type} [Inhabited A] (M : Algebra A) (s : State A)
    (bounds : RegionName → Nat) (N B : Nat) :
    TraceSafe M bounds (backward Bool.false N B) s ↔
      TraceSafe M bounds (backward Bool.true N B) s := same_safe M bounds Bool.true N B 0 Bool.false s

#axiomsClean forward_exec
#axiomsClean backward_exec
#axiomsClean forward_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.FastLayernorm.SourcePair
