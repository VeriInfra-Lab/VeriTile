import bench.tritonbench_g.cross_entropy_ops.Kernels
import bench.tritonbench_g.cross_entropy_ops.Proofs.KernelNative
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.CrossEntropyOps.SourcePair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (backward candidate : Bool) (p : Parameters) : ComputeKernel :=
  if backward then
    if candidate then backwardIndexCommutedSyntax p else backwardSyntax p
  else if candidate then forwardIndexCommutedSyntax p else forwardSyntax p

private def position (backward : Bool) : Nat := if backward then 4 else 3

/-- Isolate the changed statement from the complete readable source. -/
private def indexCode (backward candidate : Bool) (p : Parameters) : List ComputeStmt :=
  ((source backward candidate p).surfaceBody.drop (position backward)).take 1

private theorem index_run {α : Type} [Inhabited α] (M : Algebra α)
    (backward : Bool) (p : Parameters) (s : State α) :
    run M (indexCode backward Bool.false p) s = run M (indexCode backward Bool.true p) s := by
  cases backward <;> cases h : s.regs .int [] "col_block_idx"
  all_goals simp (config := { maxSteps := 1000000 }) [indexCode,position,source,
    forwardSyntax,backwardSyntax,forwardIndexCommutedSyntax,backwardIndexCommutedSyntax,
    ComputeKernel.surfaceBody,List.drop,List.take,run,step,evalExpr,evalOp_unfold,
    numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,h,Option.bind,Int.mul_comm]
  all_goals try (congr 1; funext i; simp [bop,Broadcast.leftIndex,
    Broadcast.rightIndex,Int.mul_comm])

private theorem index_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (backward candidate : Bool) (p : Parameters) (s : State α) :
    Safety.traceSafeList M bounds (indexCode backward candidate p) s := by
  cases backward <;> cases candidate
  all_goals simp [indexCode,position,source,forwardSyntax,backwardSyntax,
    forwardIndexCommutedSyntax,backwardIndexCommutedSyntax,ComputeKernel.surfaceBody,
    List.drop,List.take,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,
    Safety.safeAlg,Safety.safeOp,Safety.safeOp_unfold]
  all_goals split <;> trivial

private theorem split_source (backward candidate : Bool) (p : Parameters) :
    (source backward candidate p).surfaceBody =
      (source backward candidate p).surfaceBody.take (position backward) ++
        (indexCode backward candidate p ++
          (source backward candidate p).surfaceBody.drop (position backward+1)) := by
  cases backward <;> cases candidate <;> rfl

private theorem prefix_eq (backward : Bool) (p : Parameters) :
    (source backward Bool.false p).surfaceBody.take (position backward) =
      (source backward Bool.true p).surfaceBody.take (position backward) := by
  cases backward <;> rfl

private theorem suffix_eq (backward : Bool) (p : Parameters) :
    (source backward Bool.false p).surfaceBody.drop (position backward+1) =
      (source backward Bool.true p).surfaceBody.drop (position backward+1) := by
  cases backward <;> rfl

/-- Exact signed integer commutation preserves the entire successful or failed
source run. No floating law or input-word restriction is used. -/
theorem same_run {α : Type} [Inhabited α] (M : Algebra α)
    (backward : Bool) (p : Parameters) (s : State α) :
    exec M (source backward Bool.false p) s = exec M (source backward Bool.true p) s := by
  change run M (source backward Bool.false p).surfaceBody s =
    run M (source backward Bool.true p).surfaceBody s
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [run_append,prefix_eq,index_run,suffix_eq]

/-- The changed statement has no memory access. Identical intermediate states
therefore preserve every later source access-safety obligation. -/
theorem same_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (backward : Bool) (p : Parameters) (s : State α) :
    TraceSafe M bounds (source backward Bool.false p) s ↔
      TraceSafe M bounds (source backward Bool.true p) s := by
  have tail : ∀ t,
      Safety.traceSafeList M bounds (indexCode backward Bool.false p ++
        (source backward Bool.false p).surfaceBody.drop (position backward+1)) t ↔
      Safety.traceSafeList M bounds (indexCode backward Bool.true p ++
        (source backward Bool.true p).surfaceBody.drop (position backward+1)) t := by
    intro t
    simp only [Safety.traceSafeList_append,suffix_eq,index_run,index_safe,true_and]
  unfold TraceSafe
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [Safety.traceSafeList_append,prefix_eq,tail]

#axiomsClean same_run
#axiomsClean same_safe
end
end VeriTile.Bench.Repairs.CrossEntropyOps.SourcePair
