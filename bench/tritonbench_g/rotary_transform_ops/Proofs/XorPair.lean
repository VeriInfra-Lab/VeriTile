import bench.tritonbench_g.rotary_transform_ops.IO
import bench.tritonbench_g.rotary_transform_ops.Proofs.IOComplete
import VeriTile.Triton.Float.ReadProgramEquivalenceSyntax
import Mathlib.Data.Nat.Bitwise

namespace VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Exact adjacent partner address, independent of floating values. -/
theorem swap_xor (n : Nat) : n+((n+1)%2)*2-1 = n ^^^ 1 := by
  by_cases h : n%2=0
  · rw [Nat.xor_one_of_even (Nat.even_iff.mpr h)]
    have hp : (n+1)%2=1 := by omega
    rw [hp]
    omega
  · have odd : n%2=1 := by omega
    rw [Nat.xor_one_of_odd (Nat.odd_iff.mpr odd)]
    have hp : (n+1)%2=0 := by omega
    rw [hp]
    omega

/-- Proof views are slices of the actual complete candidate quotation. -/
def continuation (s : Shape) : List ComputeStmt :=
  ((kernel s Bool.true Bool.true).surfaceBody.drop 6).flatMap TypedExecution.ifBody

def interleavedBody (s : Shape) : List ComputeStmt :=
  ((continuation s).drop 4).flatMap (TypedExecution.branchBody Bool.true)

def splitBody (s : Shape) : List ComputeStmt :=
  ((continuation s).drop 4).flatMap (TypedExecution.branchBody Bool.false)

/-- The actual changed assignment computes the same register on every state,
including a missing rk register. No source evaluator equation is assumed. -/
theorem swap_step {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (u : State α) :
    FP.Structural.step M ((interleavedBody s).headD (.effectMarker "missing")) u =
      FP.Structural.step M ((TypedExecution.interleavedBody s).headD (.effectMarker "missing")) u := by
  simp (maxSteps := 200000) [interleavedBody,continuation,kernel,XorSource.source,
    XorSource.source64_64,TypedExecution.interleavedBody,TypedExecution.continuation,
    TypedExecution.kernel,KernelSource.source,KernelSource.source64_64,
    ComputeKernel.surfaceBody,TypedExecution.ifBody,TypedExecution.branchBody,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    numeric,natLt,numericLt,numericLe,numericEq,FP.Structural.bop,integralFloorDiv,integralMod,
    FloatDType.numericDType,FloatDType.toTileDType,ComputeDType.eraseDType,swap_xor]
  cases hr : u.regs .nat [s.blockK] "rk" with
  | none => simp [hr,Option.bind]
  | some values =>
    simp only [hr,Option.bind_some,Option.some.injEq]
    congr 1
    funext i
    simpa only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex] using
      (swap_xor (values i)).symm

/-- Only the first interleaved assignment differs; the continuation is the
same source code and consumes the proved identical successor state. -/
theorem interleaved_run {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (u : State α) :
    FP.Structural.run M (interleavedBody s) u =
      FP.Structural.run M (TypedExecution.interleavedBody s) u := by
  have tail : (interleavedBody s).drop 1 = (TypedExecution.interleavedBody s).drop 1 := by rfl
  have lhsCons : interleavedBody s =
      (interleavedBody s).headD (.effectMarker "missing") :: (interleavedBody s).drop 1 := by rfl
  have rhsCons : TypedExecution.interleavedBody s =
      (TypedExecution.interleavedBody s).headD (.effectMarker "missing") :: (TypedExecution.interleavedBody s).drop 1 := by rfl
  conv_lhs => rw [lhsCons]
  conv_rhs => rw [rhsCons]
  simp only [FP.Structural.run]

  rw [swap_step,tail]

theorem continuation_run {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (u : State α) :
    FP.Structural.run M (continuation s) u = FP.Structural.run M (TypedExecution.continuation s) u := by
  have samePrefix : (continuation s).take 4 = (TypedExecution.continuation s).take 4 := by rfl
  have tail : ∀ v : State α, FP.Structural.run M ((continuation s).drop 4) v =
      FP.Structural.run M ((TypedExecution.continuation s).drop 4) v := by
    intro v
    change FP.Structural.run M [.ifThenElse (.boolNot (.constBool s.interleaved)) (splitBody s) (interleavedBody s)] v =
      FP.Structural.run M [.ifThenElse (.boolNot (.constBool s.interleaved))
        (TypedExecution.splitBody s) (TypedExecution.interleavedBody s)] v
    rw [TypedExecution.run_if_not,TypedExecution.run_if_not]
    cases s.interleaved
    · rfl
    · exact interleaved_run M s v
  conv_lhs => rw [← List.take_append_drop 4 (continuation s)]
  conv_rhs => rw [← List.take_append_drop 4 (TypedExecution.continuation s)]
  simp only [run_append,samePrefix]
  exact congrArg (fun f => (FP.Structural.run M ((TypedExecution.continuation s).take 4) u).bind f) (funext tail)

/-- Exact full-state equality by the actual source prefix, branch assignment,
and unchanged continuation; no floating operation is rewritten. -/
theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (cu64 off64 : Bool) (u : State α) :
    FP.Structural.exec M (kernel s cu64 off64) u =
      FP.Structural.exec M (TypedExecution.kernel s cu64 off64) u := by
  have carrier : kernel s cu64 off64 = kernel s Bool.true Bool.true := by
    simp only [kernel,XorSource.signed_carrier]
  rw [carrier,TypedExecution.kernel_carrier]
  have samePrefix : (kernel s Bool.true Bool.true).surfaceBody.take 6 =
      (TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6 := by rfl
  have tail : ∀ v : State α,
      FP.Structural.run M ((kernel s Bool.true Bool.true).surfaceBody.drop 6) v =
      FP.Structural.run M ((TypedExecution.kernel s Bool.true Bool.true).surfaceBody.drop 6) v := by
    intro v
    change FP.Structural.run M [.ifThen (TypedExecution.guard s) (continuation s)] v =
      FP.Structural.run M [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] v
    rw [TypedExecution.run_if,TypedExecution.run_if]
    simp only [show FP.Structural.run M (continuation s) =
      FP.Structural.run M (TypedExecution.continuation s) from funext (continuation_run M s)]
  change FP.Structural.run M (kernel s Bool.true Bool.true).surfaceBody u =
    FP.Structural.run M (TypedExecution.kernel s Bool.true Bool.true).surfaceBody u
  conv_lhs => rw [← List.take_append_drop 6 (kernel s Bool.true Bool.true).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 6 (TypedExecution.kernel s Bool.true Bool.true).surfaceBody]
  simp only [run_append,samePrefix]
  exact congrArg (fun f => (FP.Structural.run M ((TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6) u).bind f)
    (funext tail)

#axiomsClean swap_xor
#axiomsClean swap_step
#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
