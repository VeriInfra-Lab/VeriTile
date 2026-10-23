import bench.tritonbench_g.rmsnorm_implementation.Proofs.WholeStructure
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

/-- Exact Real interpretation of each local scalar operation. -/
def realNumerics : Numerics ℝ where
  literal := id
  squareAcc := fun x a => x*x+a
  add := (·+·)
  divideByWidth := fun x width => x/(width : ℝ)
  addEpsilon := (·+·)
  div := (·/·)
  sqrt := Real.sqrt
  mul := (·*·)

private def logicalState (x : Nat → ℝ) : BlockState where
  mem := fun _ k => MemCell.real (x k)
  regs := fun _ _ _ => none
  pids := fun _ => 0

private theorem logicalState_read (x : Nat → ℝ) (k : Nat) :
    (logicalState x).readMem "X" (xColOffset (logicalState x) 0 0 1 k) = x k := by
  simp [logicalState,BlockState.readMem,xColOffset]

/-- The independently specified logical recurrence partitions the original
finite tensor sum by source lane; no interpreter or kernel result is assumed. -/
theorem accumulator_real_prefix (width : Nat) (x : Nat → ℝ) (t : Nat) (j : Fin 4096) :
    laneAccumulator realNumerics width x t j =
      rmsVarLanePrefix (logicalState x) "X" 0 0 1 width 4096 (t*4096) j := by
  induction t with
  | zero => simp [laneAccumulator,realNumerics]
  | succ t ih =>
    rw [Nat.succ_mul,rmsVarLanePrefix_step _ _ _ _ _ _ _ _ _ (by omega)]
    rw [laneAccumulator,ih,logicalState_read]
    dsimp only [realNumerics]
    split <;> simp_all [pow_two,add_comm]

private theorem sumTree_real (tree : SumTree) (x : Fin 4096 → ℝ) :
    tree.eval (·+·) x = (tree.leaves.map x).sum := by
  induction tree with
  | leaf j => simp [SumTree.eval,SumTree.leaves]
  | add a b ha hb => simp [SumTree.eval,SumTree.leaves,ha,hb,List.sum_append]

theorem reduction_real (red : Reduction) (x : Fin 4096 → ℝ) :
    red.tree.eval (·+·) x = ∑ j : Fin 4096, x j := by
  rw [sumTree_real,(red.complete.map x).sum_eq]
  rw [← List.ofFn_eq_map,List.sum_ofFn]

theorem logical_sum_real (red : Reduction) (width : Nat) (x : Nat → ℝ) :
    red.tree.eval realNumerics.add
      (laneAccumulator realNumerics width x ((width+4095)/4096)) =
      ∑ j : Fin width, (x j.val)^2 := by
  change red.tree.eval (·+·) _ = _
  rw [reduction_real]
  have bound : width ≤ ((width+4095)/4096)*4096 := by omega
  have carrier := rmsVarAccumulatorSpec_final_sum_eq_fullN
    (logicalState x) "X" 0 0 1 width 4096 (((width+4095)/4096)*4096)
    (by decide) bound
  simp only [rmsVarAccumulatorSpec,WithBot.unbotD,WithBot.recBotCoe,id_eq] at carrier
  rw [show (fun j => laneAccumulator realNumerics width x ((width+4095)/4096) j) =
      (fun j => rmsVarLanePrefix (logicalState x) "X" 0 0 1 width 4096
        (((width+4095)/4096)*4096) j) from funext (accumulator_real_prefix width x _)]
  rw [carrier]
  simp only [rmsVarFullNCarrier,logicalState_read]

/-- The non-circular Real bridge: original coordinates and the ordinary finite
sum/sqrt RMS formula, independent of either traversal and of reduction order. -/
theorem logicalValue_real (red : Reduction) (width : Nat) (eps : ℝ)
    (x w : Nat → ℝ) (k : Nat) :
    logicalValue realNumerics red width eps x w k = mathematical width eps x w k := by
  unfold logicalValue logicalStd
  rw [logical_sum_real]
  rfl

theorem mathematical_denominator_pos (width : Nat) (eps : ℝ) (positive : 0 < eps)
    (x : Nat → ℝ) :
    0 < Real.sqrt ((∑ j : Fin width, (x j.val)^2)/(width : ℝ)+eps) := by
  apply Real.sqrt_pos.2
  have squares : 0 ≤ ∑ j : Fin width, (x j.val)^2 :=
    Finset.sum_nonneg (fun j _ => sq_nonneg (x j.val))
  exact add_pos_of_nonneg_of_pos (div_nonneg squares (Nat.cast_nonneg width)) positive

/-- Whole-grid Real result, including arbitrary input/weight strides and input
aliasing. Empty dimensions have no output cells and preserve the full memory. -/
theorem real_grid_result (red : Reduction) (p : Parameters)
    (sep : Separation p) (unique : OutputUnique p) (u : Nat → ℝ)
    (order : List (Nat × Nat)) (schedule : order.Perm (programGrid p)) (memory : Memory ℝ) :
    (∀ b m k, b < p.batches → m < p.rows → k < p.width →
      executeGrid realNumerics red p u order memory p.output.allocation (outputOffset p b m k) =
        mathematical p.width p.eps
          (fun j => memory p.input.allocation (inputOffset p b m j))
          (fun j => memory p.weights.allocation (weightOffset p j)) k) ∧
    (∀ a i, Outside p a i → executeGrid realNumerics red p u order memory a i = memory a i) := by
  obtain ⟨values,frame⟩ := grid_result realNumerics red p sep unique u order schedule memory
  refine ⟨?_,frame⟩
  intro b m k hb hm hk
  rw [values b m k hb hm hk,logicalValue_real]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
