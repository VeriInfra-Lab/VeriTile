import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.RealBindings

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
open scoped BigOperators

theorem formula_agreement (c : Host.Call) (capacity : RegionBounds)
    (a b : Formula.Inputs) (agree : Agreement c capacity a b)
    (l : Formula.Launch) (bounds : AddressBounds.Valid c.layout c.regions l a capacity)
    (scale : ℝ) (i : Fin c.layout.BM) (d : Fin c.layout.D) :
    Formula.expected c.layout l a scale i.val d.val=
      Formula.expected c.layout l b scale i.val d.val := by
  have start:=agree.start _ bounds.start
  have sequence:=agree.sequence _ bounds.sequence
  have seq : Formula.sequence c.layout l a=Formula.sequence c.layout l b := by
    simp only [Formula.sequence,sequence]
  have count : Formula.count c.layout l a=Formula.count c.layout l b := by
    simp only [Formula.count,seq]
  have bn : 0<c.layout.BN := by rw [c.blockN]; decide
  have query (e : Fin c.layout.D) : Formula.query c.layout l a i.val e.val=
      Formula.query c.layout l b i.val e.val := by
    unfold Formula.query
    rw [←seq,←start]
    split_ifs with active
    · exact agree.q _ (AddressBounds.inside_toNat (bounds.query i e active))
    · rfl
  have key (j : Nat) (hj : j<Formula.count c.layout l a*c.layout.BN) (e : Fin c.layout.D) :
      Formula.key c.layout l a j e.val=Formula.key c.layout l b j e.val := by
    unfold Formula.key
    rw [←seq,←start]
    split_ifs with active
    · have hc : j/c.layout.BN<Formula.count c.layout l a := (Nat.div_lt_iff_lt_mul bn).mpr hj
      have hh:=bounds.key (j/c.layout.BN) hc ⟨j%c.layout.BN,Nat.mod_lt _ bn⟩ e
      have he : j/c.layout.BN*c.layout.BN+j%c.layout.BN=j := by simpa [Nat.mul_comm] using Nat.div_add_mod j c.layout.BN
      simp only [he] at hh
      exact agree.k _ (AddressBounds.inside_toNat (hh active))
    · rfl
  have value (j : Nat) (hj : j<Formula.count c.layout l a*c.layout.BN) :
      Formula.value c.layout l a j d.val=Formula.value c.layout l b j d.val := by
    unfold Formula.value
    rw [←seq,←start]
    split_ifs with active
    · have hc : j/c.layout.BN<Formula.count c.layout l a := (Nat.div_lt_iff_lt_mul bn).mpr hj
      have hh:=bounds.value (j/c.layout.BN) hc ⟨j%c.layout.BN,Nat.mod_lt _ bn⟩ d
      have he : j/c.layout.BN*c.layout.BN+j%c.layout.BN=j := by simpa [Nat.mul_comm] using Nat.div_add_mod j c.layout.BN
      simp only [he] at hh
      exact agree.v _ (AddressBounds.inside_toNat (hh active))
    · rfl
  have score (j : Nat) (hj : j<Formula.count c.layout l a*c.layout.BN) :
      Formula.score c.layout l a scale i.val j=Formula.score c.layout l b scale i.val j := by
    unfold Formula.score
    split_ifs
    · congr 1
      apply Finset.sum_congr rfl
      intro e _
      rw [query e,key j hj e]
    · rfl
  simp only [Formula.expected,Formula.numerator,Formula.denominator]
  congr 1 <;> apply Fintype.sum_equiv (finCongr (congrArg (fun n=>n*c.layout.BN) count)) <;> intro j
  · change _=Real.exp (Formula.score c.layout l b scale i.val j.val)*Formula.value c.layout l b j.val d.val
    rw [score j.val j.isLt,value j.val j.isLt]
  · change _=Real.exp (Formula.score c.layout l b scale i.val j.val)
    rw [score j.val j.isLt]

#axiomsClean formula_agreement
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
