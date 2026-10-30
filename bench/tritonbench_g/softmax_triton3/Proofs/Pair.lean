import bench.tritonbench_g.softmax_triton3.Proofs.PairExecution
import bench.tritonbench_g.softmax_triton2.Proofs.Pair

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
open VeriTile.Triton FP.Structural FP.Guarded
open VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Complete-memory agreement at every row, with successful executions on
all inputs. Only the selected fast path uses DIV-MUL-RCP. -/
theorem row_memory {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (model : Models R.assumptions M D)
    (ports : Ports M D) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) (hB : 0 < B) (hcount : B < 2^32) (s : State α) :
    (originalExecution M out inp mask stride n B hasMask hB s).val.mem =
      (candidateExecution M D ports out inp mask stride n B hasMask hB s).val.mem := by
  classical
  let num := numeratorValue M inp mask stride n B hasMask s
  let den := denominatorValue M inp mask stride n B hasMask s
  by_cases hc : TileShape.discreteBoolSum ⟨0,by simp⟩ Bool.false
      (fun i => ports.finite (num i)) PUnit.unit = B
  all_goals by_cases hf : ports.finite den = Bool.true
  all_goals by_cases hz : ports.atLeast (M.literal (some .fp32) .real 1) den = Bool.true
  all_goals dsimp [num,den,numeratorValue,
    denominatorValue] at hc hf hz
  · have finite := enabled_values M D ports B hcount num den ⟨hc,hf,hz⟩
    have values (i : TileIndex [B]) :
        M.binary (some .fp32) .real .div (num i) den =
        M.binary (some .fp32) .real .mul (num i)
          (M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1) den) :=
      FP.ScalarArithmetic.div_mul_rcp R M D model s _ _ (finite.1 i) finite.2.1 finite.2.2
    cases hasMask
    all_goals
      dsimp at hc hf hz
      simp [originalExecution,
        candidateExecution,numeratorValue,
        denominatorValue,dif_pos,dif_neg,hc,hf,hz,
        Bool.false_eq_true,Bool.true_eq_false,bop,Broadcast.leftIndex,
        Broadcast.rightIndex,decide_eq_true_eq,
        VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
        State.setReg_mem,ofFloat,toFloat]
      simp only [FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,
        Option.getD_some,Option.getD_none]
      congr 1
      funext mem i
      by_cases hi : i.1.val < n
      · simp only [hi,↓reduceIte]
        have eq := values i
        dsimp [num,den,numeratorValue,
          denominatorValue] at eq
        simp only [hi,↓reduceIte] at eq
        exact congrArg (fun value => fun r o =>
          if r = out ∧ o = s.pids 0*stride+i.1.val
          then Cell.mk .real value else mem r o) eq
      · simp only [hi,↓reduceIte]
  all_goals cases hasMask
  all_goals
    dsimp at hc hf hz
    simp [originalExecution,
      candidateExecution,numeratorValue,
      denominatorValue,dif_pos,dif_neg,hc,hf,hz,
      Bool.false_eq_true,Bool.true_eq_false,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,decide_eq_true_eq,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
      State.setReg_mem,ofFloat,toFloat]
    all_goals rfl

#axiomsClean row_memory
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
