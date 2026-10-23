import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardBody
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]
theorem backwardInnerExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c n : Nat) (outerMem : Memory α)
    (hcn : c+n≤p.blocks) (hBM : 0<p.blockM)
    (h : BackwardInnerInvariant p initial c (c+n) outerMem
      (backwardInnerRounds P p initial outerMem c n) current) :
    ∃ final, FP.Structural.range (backend rest P) "start_m" ((c+n)*p.blockM)
      (p.blocks*p.blockM) p.blockM (backwardInnerBody p) current=some final ∧
      BackwardInnerInvariant p initial c p.blocks outerMem
        (backwardInnerRounds P p initial outerMem c (p.blocks-c)) final := by
  have aux : ∀ fuel n (current : State α), p.blocks-(c+n)=fuel → c+n≤p.blocks →
      BackwardInnerInvariant p initial c (c+n) outerMem
        (backwardInnerRounds P p initial outerMem c n) current →
      ∃ final, FP.Structural.range (backend rest P) "start_m" ((c+n)*p.blockM)
        (p.blocks*p.blockM) p.blockM (backwardInnerBody p) current=some final ∧
        BackwardInnerInvariant p initial c p.blocks outerMem
          (backwardInnerRounds P p initial outerMem c (p.blocks-c)) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro n current he hn h
      by_cases hb : c+n<p.blocks
      · let next := backwardBodyResult P p initial current c (c+n) outerMem (backwardInnerRounds P p initial outerMem c n)
        have hi : BackwardInnerInvariant p initial c (c+(n+1)) outerMem
            (backwardInnerRounds P p initial outerMem c (n+1)) next := by
          simpa only [Nat.add_assoc] using backwardBodyInvariant P p initial current c (c+n) outerMem _ h
        obtain ⟨final,hr,hf⟩ := ih (p.blocks-(c+(n+1))) (by omega) (n+1) next rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by omega),if_pos (Nat.mul_lt_mul_of_pos_right hb hBM),
          backwardBodyExecution rest P p initial current c (c+n) outerMem _ h]
        simpa only [Nat.add_mul,Nat.one_mul,Nat.add_assoc] using hr
      · have hj : c+n=p.blocks := by omega
        have hn : n=p.blocks-c := by omega
        refine ⟨current,?_,?_⟩
        · rw [FP.Structural.range,if_neg (by omega),if_neg (by rw [hj]; exact Nat.lt_irrefl _)]
        · rw [hj,hn] at h
          exact h
  exact aux _ n current rfl hcn h

#axiomsClean backwardInnerExecution
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
