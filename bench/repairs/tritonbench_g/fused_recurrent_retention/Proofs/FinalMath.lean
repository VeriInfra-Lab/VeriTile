import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.API
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators

def mathF (s : Shape) (x : Inputs ℝ) (seed : Bool) (h k v : Nat) : ℝ :=
  mathState s x seed h k v s.T

theorem mathAdjoint_final_contribution (s : Shape) (x : Inputs ℝ) (h t k v : Nat) :
    mathAdjoint s x Bool.true h t k v=mathAdjoint s x Bool.false h t k v+
      (decay realArithmetic (h%s.H))^(s.T-1-t)*x.finalGrad h k v := by
  simp [mathAdjoint]

theorem mathDK_final_contribution (s : Shape) (x : Inputs ℝ) (h t k : Nat) :
    mathDK s x Bool.true h t k=mathDK s x Bool.false h t k+
      ∑ v∈Finset.range s.DV,((decay realArithmetic (h%s.H))^(s.T-1-t)*x.finalGrad h k v)*x.v h t v := by
  unfold mathDK
  simp_rw [mathAdjoint_final_contribution,add_mul]
  exact Finset.sum_add_distrib

theorem mathDV_final_contribution (s : Shape) (x : Inputs ℝ) (h t v : Nat) :
    mathDV s x Bool.true h t v=mathDV s x Bool.false h t v+
      ∑ k∈Finset.range s.DK,((decay realArithmetic (h%s.H))^(s.T-1-t)*x.finalGrad h k v)*x.k h t k := by
  unfold mathDV
  simp_rw [mathAdjoint_final_contribution,add_mul]
  exact Finset.sum_add_distrib

def finalOnlyWitness : Inputs ℝ where
  q := fun _ _ _ => 0
  k := fun _ _ k => if k=0 then 1 else 0
  v := fun _ _ v => if v=0 then 1 else 0
  do_ := fun _ _ _ => 0
  seed := fun _ _ _ => 0
  finalGrad := fun _ k v => if k=0 ∧ v=0 then 1 else 0

theorem finalOnlyWitness_nonzero :
    mathDK ⟨1,1,1,16,16⟩ finalOnlyWitness Bool.true 0 0 0=1 ∧
    mathDV ⟨1,1,1,16,16⟩ finalOnlyWitness Bool.true 0 0 0=1 ∧
    mathDK ⟨1,1,1,16,16⟩ finalOnlyWitness Bool.false 0 0 0=0 ∧
    mathDV ⟨1,1,1,16,16⟩ finalOnlyWitness Bool.false 0 0 0=0 := by
  norm_num [mathDK,mathDV,mathAdjoint,Legacy.suffix,
    VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.MaskedMath.decayedSuffix,finalOnlyWitness]

#axiomsClean mathAdjoint_final_contribution
#axiomsClean mathDK_final_contribution
#axiomsClean mathDV_final_contribution
#axiomsClean finalOnlyWitness_nonzero
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
