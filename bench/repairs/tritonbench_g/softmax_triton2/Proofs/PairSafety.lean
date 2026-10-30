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
  classical
  let num := SoftmaxTriton1.Rewrite.numeratorValue M inp ib ir ic n B s
  let den := SoftmaxTriton1.Rewrite.denominatorValue M inp ib ir ic n B s
  by_cases hc : TileShape.discreteBoolSum ⟨0,by simp⟩ Bool.false
      (fun i => ports.finite (num i)) PUnit.unit = B
  all_goals by_cases hf : ports.finite den = Bool.true
  all_goals by_cases hz : ports.atLeast (M.literal (some .fp32) .real 1) den = Bool.true
  all_goals
    dsimp [num,den,SoftmaxTriton1.Rewrite.numeratorValue,
      SoftmaxTriton1.Rewrite.denominatorValue] at hc hf hz
    simp (maxSteps := 1600000) [TraceSafe,guarded_reciprocal_kernel,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,FP.Structural.State.setReg,FP.Structural.evalExpr,
      FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
      FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
      FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,
      FP.Structural.numericEq,FP.Structural.numericLe,
      ComputeDType.eraseDType,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,hB,
      Region.cast,Option.bind,FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
      resolvePrecision,Function.comp_def,(show (1.0 : ℝ) = 1 by norm_num),ports.finite_bind,ports.order_bind,
      ← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,hc,hf,hz] at input output ⊢
    exact ⟨input,output⟩

/-- Original source accesses the same active input and output windows. -/
theorem original_safe {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B)
    (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B, j.val < n → ib+s.pids 0*ir+j.val*ic < bounds inp)
    (output : ∀ j : Fin B, j.val < n → ob+s.pids 0*orow+j.val*oc < bounds out) :
    TraceSafe (FP.ProfiledRewrite.engine M) bounds
      (softmax_kernel out inp ib ob ir ic orow oc n B) s := by
  simp (maxSteps := 800000) [TraceSafe,softmax_kernel,SoftmaxTriton1.softmax_kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,FP.Structural.State.setReg,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,FP.Structural.numericEq,ComputeDType.eraseDType,
    FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,hB,
    Region.cast,Option.bind,FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
    resolvePrecision,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at input output ⊢
  exact ⟨input,output⟩

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
