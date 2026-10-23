import bench.tritonbench_g.rmsnorm_triton.Proofs.WholeStructure
import bench.tritonbench_g.rmsnorm_triton.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Exact mathematical interpretation of local native instructions. This is
an independent Real specification, not a claim that FP32 operations are exact. -/
def realNumerics : Numerics ℝ where
  literal := id
  mulFTZ := (·*·)
  add := (·+·)
  div := (·/·)
  rsqrt := fun x => (Real.sqrt x)⁻¹
  mul := (·*·)

theorem squareAcc_real (initialRightZero : Bool) (t : Nat) (x acc : ℝ) :
    realNumerics.squareAcc initialRightZero t x acc = acc+x^2 := by
  simp only [Numerics.squareAcc,realNumerics,id_eq]
  split_ifs <;> ring

/-- The logical recurrence partitions the independent finite sum into its
original block/lane coordinates. The established reblocking lemma is reused. -/
theorem accumulator_real_prefix (initialRightZero : Bool) (width : Nat)
    (x : Nat → ℝ) (t : Nat) (j : Fin 1024) :
    laneAccumulator realNumerics initialRightZero width x t j =
      ∑ b : Fin t, if b.val*1024+j.val < width then (x (b.val*1024+j.val))^2 else 0 := by
  induction t with
  | zero => simp [laneAccumulator,realNumerics]
  | succ t ih =>
    rw [laneAccumulator,squareAcc_real,ih,Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc,Fin.val_last]
    split_ifs <;> simp_all [realNumerics]

private theorem sumTree_real (tree : SumTree) (x : Fin 1024 → ℝ) :
    tree.eval (·+·) x = (tree.leaves.map x).sum := by
  induction tree with
  | leaf j => simp [SumTree.eval,SumTree.leaves]
  | add a b ha hb => simp [SumTree.eval,SumTree.leaves,ha,hb,List.sum_append]

theorem reduction_real (red : Reduction) (x : Fin 1024 → ℝ) :
    red.tree.eval (·+·) x = ∑ j : Fin 1024, x j := by
  rw [sumTree_real,(red.complete.map x).sum_eq]
  rw [← List.ofFn_eq_map,List.sum_ofFn]

theorem logical_sum_real (red : Reduction) (width : Nat) (x : Nat → ℝ) :
    red.tree.eval realNumerics.add
      (laneAccumulator realNumerics red.initialRightZero width x ((width+1023)/1024)) =
      ∑ j : Fin width, (x j.val)^2 := by
  change red.tree.eval (·+·) _ = _
  rw [reduction_real]
  simp_rw [accumulator_real_prefix]
  exact Full.MathHeart.sum_sq_mean 1024 ((width+1023)/1024) width
    (by decide) (by omega) (fun k => (x k)^2)

/-- Original logical tensors and the ordinary RMS formula. Associativity and
commutativity appear only in this Real bridge, never in the native FP theorem. -/
theorem logicalValue_real (red : Reduction) (width : Nat) (eps : ℝ)
    (x w : Nat → ℝ) (k : Nat) :
    logicalValue realNumerics red width eps x w k = mathematical width eps x w k := by
  unfold logicalValue logicalStd
  rw [logical_sum_real]
  by_cases h : width = 1
  · subst width
    simp [Numerics.addEpsilon,Numerics.divideByWidth,realNumerics,mathematical,div_eq_mul_inv,mul_comm]
  · simp [Numerics.addEpsilon,Numerics.divideByWidth,realNumerics,mathematical,h,div_eq_mul_inv,mul_comm]

theorem mathematical_denominator_pos (width : Nat) (eps : ℝ) (positive : 0 < eps)
    (x : Nat → ℝ) :
    0 < Real.sqrt ((∑ j : Fin width, (x j.val)^2)/(width : ℝ)+eps) := by
  apply Real.sqrt_pos.2
  have squares : 0 ≤ ∑ j : Fin width, (x j.val)^2 :=
    Finset.sum_nonneg (fun j _ => sq_nonneg (x j.val))
  exact add_pos_of_nonneg_of_pos (div_nonneg squares (Nat.cast_nonneg width)) positive

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
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
