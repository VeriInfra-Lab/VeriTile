import bench.repairs.tritonbench_g.softmax_triton2.Proofs.GuardedReciprocal
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open VeriTile.Triton FP.Structural FP.Guarded
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The candidate adds pure runtime checks and arithmetic only. Every actual
memory access retains the original source window and mask in either branch. -/
theorem candidate_safe {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (ports : Ports M D) (out inp : RegionName)
    (ib ob ir ic orow oc n B : Nat) (hB : 0 < B)
    (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B, j.val < n → ib+s.pids 0*ir+j.val*ic < bounds inp)
    (output : ∀ j : Fin B, j.val < n → ob+s.pids 0*orow+j.val*oc < bounds out) :
    TraceSafe (FP.ProfiledRewrite.engine M) bounds
      (guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B) s := by
  simp (maxSteps := 800000) [TraceSafe,guarded_reciprocal_kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericEq,FP.Structural.numericLe,
    ComputeDType.eraseDType,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,hB,
    Region.cast,Option.bind,FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
    resolvePrecision,ports.finite_bind,ports.order_bind] at input output ⊢
  split_ifs <;>
    simp (maxSteps := 800000) [Option.bind,FP.Structural.run,FP.Structural.step,
      FP.Structural.evalExpr,FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
      FP.Structural.store,FP.Structural.State.setReg,FP.Structural.numeric,FP.Structural.natLt,
      FP.Structural.numericEq,FP.Structural.numericLe,FP.Structural.bop,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      Safety.precisionContext,ComputeDType.eraseDType,Region.cast,
      FP.Structural.evalOffsets,FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
      resolvePrecision,TileShape.axisDim,hB,ports.finite_bind,ports.order_bind]
  all_goals tauto

/-- Original source accesses the same active input and output windows. -/
theorem original_safe {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B)
    (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B, j.val < n → ib+s.pids 0*ir+j.val*ic < bounds inp)
    (output : ∀ j : Fin B, j.val < n → ob+s.pids 0*orow+j.val*oc < bounds out) :
    TraceSafe (FP.ProfiledRewrite.engine M) bounds
      (softmax_kernel out inp ib ob ir ic orow oc n B) s := by
  simp (maxSteps := 800000) [TraceSafe,softmax_kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericEq,ComputeDType.eraseDType,
    FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,hB,
    Region.cast,Option.bind,FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
    resolvePrecision] at input output ⊢
  tauto

/-- Every exterior typed cell is preserved by the complete original source. -/
theorem original_frame {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B)
    (s : State α) (r : RegionName) (a : Nat)
    (miss : ∀ i : Fin B, i.val < n → ¬(out=r ∧ ob+s.pids 0*orow+i.val*oc=a)) :
    (originalExecution M out inp ib ob ir ic orow oc n B hB s).val.mem r a = s.mem r a := by
  simp only [originalExecution,SoftmaxTriton1.Rewrite.originalExecution]
  apply State.masked_scatter_frame
  by_cases eq : r = out
  · right
    intro i _ hi off
    exact miss i.1 hi ⟨eq.symm,off.symm⟩
  · exact Or.inl eq

#axiomsClean original_safe
#axiomsClean original_frame
#axiomsClean candidate_safe
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
