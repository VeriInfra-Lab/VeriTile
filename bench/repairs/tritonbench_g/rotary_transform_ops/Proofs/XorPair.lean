import bench.repairs.tritonbench_g.rotary_transform_ops.IO
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.IOComplete
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

/-- Proof-only selection of the caller's immutable branch. -/
def sourceBody (s : Shape) : List ComputeStmt :=
  SourceBlocks.body s.varlen (kernel s Bool.true Bool.true)

theorem run_candidate_body {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (u : State α) :
    FP.Structural.run M (kernel s Bool.true Bool.true).surfaceBody u = FP.Structural.run M (sourceBody s) u := by
  have h := SourceBlocks.run_constant_branch M s.varlen
    ((kernel s Bool.true Bool.true).surfaceBody.take 4)
    (((kernel s Bool.true Bool.true).surfaceBody.drop 4).flatMap (SourceBlocks.select Bool.false))
    (((kernel s Bool.true Bool.true).surfaceBody.drop 4).flatMap (SourceBlocks.select Bool.true)) u
  cases ht : s.offsetTensor <;>
    simpa only [sourceBody,SourceBlocks.body,kernel,XorSource.source,XorSource.xor,XorSource.xor64_64,
      XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap_cons,List.flatMap_nil,SourceBlocks.select,List.append_nil,
      ht,Bool.false_eq_true,if_false,if_true] using h

theorem candidate_body_safe_iff {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (bounds : RegionName → Nat) (u : State α) :
    TraceSafe M bounds (kernel s Bool.true Bool.true) u ↔
      Safety.traceSafeList M bounds (sourceBody s) u := by
  have h := SourceBlocks.safe_constant_branch M bounds s.varlen
    ((kernel s Bool.true Bool.true).surfaceBody.take 4)
    (((kernel s Bool.true Bool.true).surfaceBody.drop 4).flatMap (SourceBlocks.select Bool.false))
    (((kernel s Bool.true Bool.true).surfaceBody.drop 4).flatMap (SourceBlocks.select Bool.true)) u
  cases ht : s.offsetTensor <;>
    simpa only [TraceSafe,sourceBody,SourceBlocks.body,kernel,XorSource.source,XorSource.xor,XorSource.xor64_64,
      XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap_cons,List.flatMap_nil,SourceBlocks.select,List.append_nil,
      ht,Bool.false_eq_true,if_false,if_true] using h

#axiomsClean run_candidate_body
#axiomsClean candidate_body_safe_iff

/-- Proof views are slices of the actual complete candidate quotation. -/
def continuation (s : Shape) : List ComputeStmt :=
  ((sourceBody s).drop (SourceBlocks.setupCount s.varlen)).flatMap TypedExecution.ifBody

def interleavedBody (s : Shape) : List ComputeStmt :=
  ((continuation s).drop 4).flatMap (TypedExecution.branchBody Bool.true)

def splitBody (s : Shape) : List ComputeStmt :=
  ((continuation s).drop 4).flatMap (TypedExecution.branchBody Bool.false)


theorem same_prefix (s : Shape) :
    (continuation s).take 4 = (TypedExecution.continuation s).take 4 := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem same_split (s : Shape) :
    splitBody s = TypedExecution.splitBody s := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem interleaved_tail (s : Shape) :
    (interleavedBody s).drop 1 = (TypedExecution.interleavedBody s).drop 1 := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem interleaved_cons (s : Shape) :
    interleavedBody s = (interleavedBody s).headD (.effectMarker "missing") :: (interleavedBody s).drop 1 := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem original_interleaved_cons (s : Shape) :
    TypedExecution.interleavedBody s = (TypedExecution.interleavedBody s).headD (.effectMarker "missing") :: (TypedExecution.interleavedBody s).drop 1 := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem continuation_tail_code (s : Shape) :
    (continuation s).drop 4 = [.ifThenElse (.boolNot (.constBool s.interleaved)) (splitBody s) (interleavedBody s)] := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem original_tail_code (s : Shape) :
    (TypedExecution.continuation s).drop 4 = [.ifThenElse (.boolNot (.constBool s.interleaved)) (TypedExecution.splitBody s) (TypedExecution.interleavedBody s)] := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

/-- The actual changed assignment computes the same register on every state,
including a missing rk register. No source evaluator equation is assumed. -/
theorem swap_step {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (u : State α) :
    FP.Structural.step M ((interleavedBody s).headD (.effectMarker "missing")) u =
      FP.Structural.step M ((TypedExecution.interleavedBody s).headD (.effectMarker "missing")) u := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor
  all_goals
   simp (maxSteps := 200000) [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
    XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
    KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,hv,ht,interleavedBody,continuation,kernel,XorSource.source,
    XorSource.source64_64,TypedExecution.interleavedBody,TypedExecution.continuation,
    TypedExecution.kernel,KernelSource.source,KernelSource.source64_64,
    ComputeKernel.surfaceBody,TypedExecution.ifBody,TypedExecution.branchBody,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    numeric,natLt,numericLt,numericLe,numericEq,FP.Structural.bop,integralFloorDiv,integralMod,
    FloatDType.numericDType,FloatDType.toTileDType,ComputeDType.eraseDType,swap_xor]
  all_goals
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
  have tail := interleaved_tail s
  have lhsCons := interleaved_cons s
  have rhsCons := original_interleaved_cons s
  conv_lhs => rw [lhsCons]
  conv_rhs => rw [rhsCons]
  simp only [FP.Structural.run]

  rw [swap_step,tail]

theorem continuation_run {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (u : State α) :
    FP.Structural.run M (continuation s) u = FP.Structural.run M (TypedExecution.continuation s) u := by
  have samePrefix := same_prefix s
  have tail : ∀ v : State α, FP.Structural.run M ((continuation s).drop 4) v =
      FP.Structural.run M ((TypedExecution.continuation s).drop 4) v := by
    intro v
    rw [continuation_tail_code,original_tail_code]
    rw [TypedExecution.run_if_not,TypedExecution.run_if_not]
    cases s.interleaved
    · simp only [same_split,Bool.false_eq_true,if_false]
    · exact interleaved_run M s v
  conv_lhs => rw [← List.take_append_drop 4 (continuation s)]
  conv_rhs => rw [← List.take_append_drop 4 (TypedExecution.continuation s)]
  simp only [run_append,samePrefix]
  exact congrArg (fun f => (FP.Structural.run M ((TypedExecution.continuation s).take 4) u).bind f) (funext tail)

theorem original_body_split (s : Shape) :
    TypedExecution.sourceBody s =
      (TypedExecution.sourceBody s).take (SourceBlocks.setupCount s.varlen) ++
        [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] := by
  rw [← TypedExecution.source_tail]
  exact (List.take_append_drop _ _).symm

theorem candidate_body_split (s : Shape) :
    sourceBody s =
      (TypedExecution.sourceBody s).take (SourceBlocks.setupCount s.varlen) ++
        [.ifThen (TypedExecution.guard s) (continuation s)] := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [TypedExecution.guard,TypedExecution.guardOf,sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      continuation,TypedExecution.continuation,interleavedBody,TypedExecution.interleavedBody,
      splitBody,TypedExecution.splitBody,kernel,TypedExecution.kernel,
      XorSource.source,XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
      KernelSource.source,KernelSource.source64_64,KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,
      TypedExecution.ifBody,TypedExecution.branchBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

/-- Exact full-state equality by the actual source prefix, branch assignment,
and unchanged continuation; no floating operation is rewritten. -/
theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (cu64 off64 : Bool) (u : State α) :
    FP.Structural.exec M (kernel s cu64 off64) u =
      FP.Structural.exec M (TypedExecution.kernel s cu64 off64) u := by
  have carrier : kernel s cu64 off64 = kernel s Bool.true Bool.true := by
    simp only [kernel,XorSource.signed_carrier]
  rw [carrier,TypedExecution.kernel_carrier]
  change FP.Structural.run M (kernel s Bool.true Bool.true).surfaceBody u =
    FP.Structural.run M (TypedExecution.kernel s Bool.true Bool.true).surfaceBody u
  rw [run_candidate_body,TypedExecution.run_source_body,original_body_split,candidate_body_split,
    run_append,run_append]
  have tail : ∀ v : State α,
      FP.Structural.run M [.ifThen (TypedExecution.guard s) (continuation s)] v =
      FP.Structural.run M [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] v := by
    intro v
    rw [TypedExecution.run_if,TypedExecution.run_if]
    simp only [show FP.Structural.run M (continuation s) =
      FP.Structural.run M (TypedExecution.continuation s) from funext (continuation_run M s)]
  exact congrArg (fun f => (FP.Structural.run M ((TypedExecution.sourceBody s).take (SourceBlocks.setupCount s.varlen)) u).bind f)
    (funext tail)

#axiomsClean swap_xor
#axiomsClean swap_step
#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
