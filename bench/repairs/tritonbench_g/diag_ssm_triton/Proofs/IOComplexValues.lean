import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOValues

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem masked_complex_read {B : Nat} (u : BlockState) (r : RegionName)
    (off : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool) (j : TileIndex [B])
    (hm : mask j=Bool.true) :
    ComplexForwardStep.input u r off mask j=ComplexMemory.read u r (off j) := by
  simp only [ComplexForwardStep.input,hm,↓reduceIte,ComplexMemory.read]

theorem forward_complex_bindings (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.true p length).Ctx) (hp : (forwardIO Bool.true p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    ComplexForwardLoop.initial p u ⟨j,PUnit.unit⟩=forwardComplexInputs p length v 0 j.val ∧
    ComplexForwardLoop.transition p u ⟨j,PUnit.unit⟩=forwardComplexInputs p length v 1 j.val ∧
    (∀ q, q<length → ComplexForwardLoop.input p u q ⟨j,PUnit.unit⟩=
      forwardComplexInputs p length v 2 (q*p.B+j.val)) := by
  have hm : ComplexForwardInit.active u p.batch p.dim p.B ⟨j,PUnit.unit⟩=Bool.true := decide_eq_true ha
  constructor
  · exact (masked_complex_read u p.s _ _ _ hm).trans
      (forward_complex_pin p length u v hp 0 j.val j.isLt ha)
  constructor
  · exact (masked_complex_read u p.l (fun i => ComplexForwardInit.columns u p.B i % p.dim) _ _ hm).trans
      (forward_complex_pin p length u v hp 1 j.val j.isLt ha)
  · intro q hq
    have h := forward_complex_pin p length u v hp 2 (pair (⟨q,hq⟩ : Fin length) j).val
      (pair (⟨q,hq⟩ : Fin length) j).isLt (by simpa [forwardLane] using ha)
    dsimp only [forwardBuffer,forwardAddress] at h
    simp only [streamAddress] at h
    simp only [pair_div,pair_mod] at h
    exact (masked_complex_read u p.x _ _ _ hm).trans h

theorem forward_complex_value (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.true p length).Ctx) (hp : (forwardIO Bool.true p length).RegionalInputs u v)
    (t : Fin length) (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    ComplexForwardLoop.state p u (t.val+1) ⟨j,PUnit.unit⟩ =
      forwardFormula p.B (forwardComplexInputs p length v) (pair t j).val := by
  obtain ⟨hs,hl,hx⟩ := forward_complex_bindings p length u v hp j ha
  simp only [ComplexForwardLoop.state,forwardFormula,pair_div,pair_mod,hs,hl]
  apply ReferenceCongruence.forward_prefix
  intro q hq
  exact hx q (by have := t.isLt; omega)

theorem backward_complex_bindings (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.true p length).Ctx) (hp : (backwardIO Bool.true p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    ComplexBackwardLoop.transition p u ⟨j,PUnit.unit⟩=backwardComplexInputs p length v 1 j.val ∧
    (∀ q, q<length → ComplexBackwardLoop.gradOutput p u q ⟨j,PUnit.unit⟩=
      backwardComplexInputs p length v 3 (q*p.B+j.val)) ∧
    (0<length → ComplexBackwardLoop.initial p u ⟨j,PUnit.unit⟩=backwardComplexInputs p length v 0 j.val) ∧
    (∀ q, q+1<length → ComplexBackwardLoop.output p u q ⟨j,PUnit.unit⟩=
      backwardComplexInputs p length v 2 (q*p.B+j.val)) := by
  have hm : ComplexForwardInit.active u p.batch p.dim p.B ⟨j,PUnit.unit⟩=Bool.true := decide_eq_true ha
  refine ⟨(masked_complex_read u p.l (fun i => ComplexForwardInit.columns u p.B i % p.dim) _ _ hm).trans
    (backward_complex_pin p length u v hp 1 j.val j.isLt (by simp) ha),?_,?_,?_⟩
  · intro q hq
    have h := backward_complex_pin p length u v hp 3 (pair (⟨q,hq⟩ : Fin length) j).val
      (pair (⟨q,hq⟩ : Fin length) j).isLt (by simp) (by simpa [backwardLane] using ha)
    dsimp only [backwardBuffer,backwardAddress] at h
    simp only [streamAddress] at h
    simp only [pair_div,pair_mod] at h
    exact (masked_complex_read u p.gy _ _ _ hm).trans h
  · intro hz
    exact (masked_complex_read u p.s _ _ _ hm).trans
      (backward_complex_pin p length u v hp 0 j.val j.isLt (by omega) ha)
  · intro q hq
    have hq' : q<length-1 := by omega
    have h := backward_complex_pin p length u v hp 2 (pair (⟨q,hq'⟩ : Fin (length-1)) j).val
      (pair (⟨q,hq'⟩ : Fin (length-1)) j).isLt (by simp) (by simpa [backwardLane] using ha)
    dsimp only [backwardBuffer,backwardAddress] at h
    simp only [streamAddress] at h
    simp only [pair_div,pair_mod] at h
    exact (masked_complex_read u p.y _ _ _ hm).trans h

theorem backward_complex_carry (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.true p length).Ctx) (hp : (backwardIO Bool.true p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) (k : Nat) (hk : k≤length) :
    Reference.reverseCarry length (starRingEnd ℂ (ComplexBackwardLoop.transition p u ⟨j,PUnit.unit⟩))
      (fun q => ComplexBackwardLoop.gradOutput p u q ⟨j,PUnit.unit⟩) k =
      Reference.reverseCarry length (starRingEnd ℂ (backwardComplexInputs p length v 1 j.val))
        (fun q => backwardComplexInputs p length v 3 (q*p.B+j.val)) k := by
  obtain ⟨hl,hg,_,_⟩ := backward_complex_bindings p length u v hp j ha
  rw [hl]
  exact ReferenceCongruence.carry_prefix length _ _ _ k hk hg

theorem backward_complex_gx (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.true p length).Ctx) (hp : (backwardIO Bool.true p length).RegionalInputs u v)
    (t : Fin length) (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    Reference.inputGradient length (starRingEnd ℂ (ComplexBackwardLoop.transition p u ⟨j,PUnit.unit⟩))
      (fun q => ComplexBackwardLoop.gradOutput p u q ⟨j,PUnit.unit⟩) t.val =
      backwardFormula (starRingEnd ℂ) p.B length (backwardComplexInputs p length v) 0 (pair t j).val := by
  obtain ⟨_,hg,_,_⟩ := backward_complex_bindings p length u v hp j ha
  simp only [backwardFormula,backwardOutLane,Fin.val_zero,↓reduceIte,pair_mod,pair_div,Reference.inputGradient]
  rw [backward_complex_carry p length u v hp j ha _ (by omega),hg t.val t.isLt]

theorem backward_complex_gl (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.true p length).Ctx) (hp : (backwardIO Bool.true p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    Reference.transitionGradient length (starRingEnd ℂ (ComplexBackwardLoop.transition p u ⟨j,PUnit.unit⟩))
      (starRingEnd ℂ (ComplexBackwardLoop.initial p u ⟨j,PUnit.unit⟩))
      (fun q => starRingEnd ℂ (ComplexBackwardLoop.output p u q ⟨j,PUnit.unit⟩))
      (fun q => ComplexBackwardLoop.gradOutput p u q ⟨j,PUnit.unit⟩) length =
      backwardFormula (starRingEnd ℂ) p.B length (backwardComplexInputs p length v) 2 j.val := by
  by_cases hz : length=0
  · subst length; rfl
  · obtain ⟨hl,hg,hs,hy⟩ := backward_complex_bindings p length u v hp j ha
    rw [hl,hs (by omega)]
    exact ReferenceCongruence.transition_prefix length _ _ _ _ _ _ length (le_refl _)
      (fun q hq => congrArg (starRingEnd ℂ) (hy q hq)) hg

end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
