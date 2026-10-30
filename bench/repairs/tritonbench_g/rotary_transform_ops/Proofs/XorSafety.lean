import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.XorPair

namespace VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem swap_safe {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (bounds : RegionName → Nat) (u : State α) :
    Safety.stepSafe M bounds ((interleavedBody s).headD (.effectMarker "missing")) u ↔
      Safety.stepSafe M bounds ((TypedExecution.interleavedBody s).headD (.effectMarker "missing")) u := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
   simp (maxSteps := 200000) [sourceBody,TypedExecution.sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
    XorSource.xor,XorSource.xor64_64,XorSource.xor64_64_tensor,XorSource.xor64_32_scalar,
    KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,hv,ht,interleavedBody,continuation,kernel,XorSource.source,
    XorSource.source64_64,TypedExecution.interleavedBody,TypedExecution.continuation,
    TypedExecution.kernel,KernelSource.source,KernelSource.source64_64,
    ComputeKernel.surfaceBody,TypedExecution.ifBody,TypedExecution.branchBody,
    Safety.stepSafe_unfold,Safety.safeAlg_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold]

private theorem interleaved_safe {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (bounds : RegionName → Nat) (u : State α) :
    Safety.traceSafeList M bounds (interleavedBody s) u ↔
      Safety.traceSafeList M bounds (TypedExecution.interleavedBody s) u := by
  have tail := interleaved_tail s
  conv_lhs => rw [interleaved_cons]
  conv_rhs => rw [original_interleaved_cons]
  simp only [Safety.traceSafeList_cons,swap_safe,swap_step,tail]

private theorem continuation_safe {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (bounds : RegionName → Nat) (u : State α) :
    Safety.traceSafeList M bounds (continuation s) u ↔
      Safety.traceSafeList M bounds (TypedExecution.continuation s) u := by
  have samePrefix := same_prefix s
  have tail : ∀ v : State α, Safety.traceSafeList M bounds ((continuation s).drop 4) v ↔
      Safety.traceSafeList M bounds ((TypedExecution.continuation s).drop 4) v := by
    intro v
    rw [continuation_tail_code,original_tail_code]
    have sameSplit := same_split s
    simp only [sameSplit]
    rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
    simp only [Safety.traceSafeList_nil,implies_true,forall_const,and_true]
    rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
    simp only [Safety.safeOp_unfold,FP.Structural.evalOp_unfold]
    cases hi : s.interleaved <;>
      simp only [hi,Bool.not_false,Bool.not_true,Bool.false_eq_true,if_false,if_true,
        true_and,interleaved_safe]
  conv_lhs => rw [← List.take_append_drop 4 (continuation s)]
  conv_rhs => rw [← List.take_append_drop 4 (TypedExecution.continuation s)]
  simp only [Safety.traceSafeList_append,samePrefix,tail]

/-- The integer-index assignment has no memory access and computes identical
successor registers. Source safety follows the same prefix and branches. -/
theorem safety_iff {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (cu64 off64 : Bool) (bounds : RegionName → Nat) (u : State α) :
    TraceSafe M bounds (kernel s cu64 off64) u ↔
      TraceSafe M bounds (TypedExecution.kernel s cu64 off64) u := by
  have carrier : kernel s cu64 off64 = kernel s Bool.true Bool.true := by
    simp only [kernel,XorSource.signed_carrier]
  rw [carrier,TypedExecution.kernel_carrier]
  rw [candidate_body_safe_iff,SourceSafetyTools.source_body_safe_iff,
    original_body_split,candidate_body_split]
  have tail : ∀ v : State α,
      Safety.traceSafeList M bounds [.ifThen (TypedExecution.guard s) (continuation s)] v ↔
      Safety.traceSafeList M bounds [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] v := by
    intro v
    rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
    simp only [Safety.traceSafeList_nil,implies_true,forall_const,and_true]
    rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
    simp only [continuation_safe]
  simp only [Safety.traceSafeList_append,tail]

#axiomsClean safety_iff
end
end VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
