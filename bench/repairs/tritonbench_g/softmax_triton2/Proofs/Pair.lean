import bench.repairs.tritonbench_g.softmax_triton2.Proofs.GuardedReciprocal

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open VeriTile.Triton FP.Structural FP.Guarded
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Runtime checks establish all the local numerical guards, without a
restriction on the original row's input words. -/
theorem enabled_values {α : Type} (M : Algebra α) (D : Domain α) (ports : Ports M D)
    (B : Nat) (hB : B < 2^32) (num : Values α .real [B]) (den : α)
    (h : enabled M D ports B num den) :
    (∀ i, D .finite (num i)) ∧ D .finite den ∧ D .nonzero den := by
  have count := h.1
  change TileShape.boolCount32 (fun i => ports.finite (num (i,PUnit.unit))) = B at count
  have all := count_all _ hB count
  refine ⟨?_, ports.finite_sound den h.2.1, ports.nonzero_sound den h.2.2⟩
  intro i
  cases i with | mk i u => cases u; exact ports.finite_sound _ (all i)

/-- Complete-memory agreement at every row, with successful executions on
all inputs. Only the selected fast path uses DIV-MUL-RCP. -/
theorem row_memory {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (model : Models R.assumptions M D)
    (ports : Ports M D) (out inp : RegionName)
    (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (hcount : B < 2^32) (s : State α) :
    (originalExecution M out inp ib ob ir ic orow oc n B hB s).val.mem =
      (candidateExecution M D ports out inp ib ob ir ic orow oc n B hB s).val.mem := by
  classical
  let num := SoftmaxTriton1.Rewrite.numeratorValue M inp ib ir ic n B s
  let den := SoftmaxTriton1.Rewrite.denominatorValue M inp ib ir ic n B s
  by_cases hc : TileShape.discreteBoolSum ⟨0,by simp⟩ Bool.false
      (fun i => ports.finite (num i)) PUnit.unit = B
  all_goals by_cases hf : ports.finite den = Bool.true
  all_goals by_cases hz : ports.atLeast (M.literal (some .fp32) .real 1) den = Bool.true
  all_goals dsimp [num,den,SoftmaxTriton1.Rewrite.numeratorValue,
    SoftmaxTriton1.Rewrite.denominatorValue] at hc hf hz
  · have finite := enabled_values M D ports B hcount num den ⟨hc,hf,hz⟩
    have values (i : TileIndex [B]) :
        M.binary (some .fp32) .real .div (num i) den =
        M.binary (some .fp32) .real .mul (num i)
          (M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1) den) :=
      FP.ScalarArithmetic.div_mul_rcp R M D model s _ _ (finite.1 i) finite.2.1 finite.2.2
    simp [originalExecution,SoftmaxTriton1.Rewrite.originalExecution,
      candidateExecution,SoftmaxTriton1.Rewrite.numeratorValue,
      SoftmaxTriton1.Rewrite.denominatorValue,dif_pos,dif_neg,hc,hf,hz,
      Bool.false_eq_true,Bool.true_eq_false,bop,Broadcast.leftIndex,
      Broadcast.rightIndex,decide_eq_true_eq,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
      State.setReg_mem]
    simp only [FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,
      Option.getD_some,Option.getD_none]
    congr 1
    funext mem i
    by_cases hi : i.1.val < n
    · simp only [hi,↓reduceIte]
      have eq := values i
      dsimp [num,den,SoftmaxTriton1.Rewrite.numeratorValue,
        SoftmaxTriton1.Rewrite.denominatorValue] at eq
      simp only [hi,↓reduceIte] at eq
      exact congrArg (fun value => fun r o =>
        if r = out ∧ o = ob+s.pids 0*orow+i.1.val*oc
        then Cell.mk .real value else mem r o) eq
    · simp only [hi,↓reduceIte]
  all_goals
    simp [originalExecution,SoftmaxTriton1.Rewrite.originalExecution,
      candidateExecution,SoftmaxTriton1.Rewrite.numeratorValue,
      SoftmaxTriton1.Rewrite.denominatorValue,dif_pos,dif_neg,hc,hf,hz,
      Bool.false_eq_true,Bool.true_eq_false,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,decide_eq_true_eq,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
      State.setReg_mem]
    all_goals rfl

#axiomsClean enabled_values
#axiomsClean row_memory
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
