import bench.tritonbench_g.diag_ssm_triton.Proofs.IOBindings
import bench.tritonbench_g.diag_ssm_triton.Proofs.RegionCorrectness
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ReferenceCongruence

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem forward_real_pin (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.false p length).Ctx) (hp : (forwardIO Bool.false p length).RegionalInputs u v)
    (role : Fin 3) (k : Nat) (hk : k<forwardArity p.B length role)
    (ha : active p.batch p.dim p.B (u.pids 0) (forwardLane p.B role k)) :
    u.readMem (forwardBuffer p role) (forwardAddress p (u.pids 0) role k)=
      forwardRealInputs p length v role k := by
  simpa [components,forward_component_real] using forward_pin Bool.false p length u v hp role (0 : Fin 1) k hk ha

theorem backward_real_pin (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.false p length).Ctx) (hp : (backwardIO Bool.false p length).RegionalInputs u v)
    (role : Fin 4) (k : Nat) (hk : k<backwardArity p.B length role) (hz : role.val=0 → 0<length)
    (ha : active p.batch p.dim p.B (u.pids 0) (backwardLane p.B role k)) :
    u.readMem (backwardBuffer p role) (backwardAddress p (u.pids 0) role k)=
      backwardRealInputs p length v role k := by
  simpa [components,backward_component_real] using backward_pin Bool.false p length u v hp role (0 : Fin 1) k hk hz ha

theorem forward_complex_pin (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.true p length).Ctx) (hp : (forwardIO Bool.true p length).RegionalInputs u v)
    (role : Fin 3) (k : Nat) (hk : k<forwardArity p.B length role)
    (ha : active p.batch p.dim p.B (u.pids 0) (forwardLane p.B role k)) :
    ComplexMemory.read u (forwardBuffer p role) (forwardAddress p (u.pids 0) role k)=
      forwardComplexInputs p length v role k := by
  apply Complex.ext
  · simpa [ComplexMemory.read,components,forward_component_complex_re] using
      forward_pin Bool.true p length u v hp role (0 : Fin 2) k hk ha
  · simpa [ComplexMemory.read,components,forward_component_complex_im] using
      forward_pin Bool.true p length u v hp role (1 : Fin 2) k hk ha

theorem backward_complex_pin (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.true p length).Ctx) (hp : (backwardIO Bool.true p length).RegionalInputs u v)
    (role : Fin 4) (k : Nat) (hk : k<backwardArity p.B length role) (hz : role.val=0 → 0<length)
    (ha : active p.batch p.dim p.B (u.pids 0) (backwardLane p.B role k)) :
    ComplexMemory.read u (backwardBuffer p role) (backwardAddress p (u.pids 0) role k)=
      backwardComplexInputs p length v role k := by
  apply Complex.ext
  · simpa [ComplexMemory.read,components,backward_component_complex_re] using
      backward_pin Bool.true p length u v hp role (0 : Fin 2) k hk hz ha
  · simpa [ComplexMemory.read,components,backward_component_complex_im] using
      backward_pin Bool.true p length u v hp role (1 : Fin 2) k hk hz ha

theorem real_forward_reference (p : ForwardLayout) (u : BlockState) (j : Fin p.B) (t : Nat) :
    RealReference.diagSsmStateAfter u p.s p.x p.l p.batch p.dim p.B j t =
      Reference.forwardState (u.readMem p.s (column p.B (u.pids 0) j.val))
        (u.readMem p.l (column p.B (u.pids 0) j.val%p.dim))
        (fun q => u.readMem p.x (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val)) t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [RealReference.diagSsmStateAfter,Reference.forwardState,ih]; rfl

theorem real_carry_reference (p : BackwardLayout) (u : BlockState) (j : Fin p.B) (length k : Nat) :
    RealReference.diagSsmBackwardGradSAfter u p.l p.gy p.batch p.dim p.B length j k =
      Reference.reverseCarry length (u.readMem p.l (column p.B (u.pids 0) j.val%p.dim))
        (fun q => u.readMem p.gy (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val)) k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [RealReference.diagSsmBackwardGradSAfter,Reference.reverseCarry,ih]; rfl

theorem real_previous_reference (p : BackwardLayout) (u : BlockState) (j : Fin p.B) (length k : Nat) :
    RealReference.diagSsmBackwardPrevState u p.s p.y p.batch p.dim p.B length k j =
      Reference.previousState (u.readMem p.s (column p.B (u.pids 0) j.val))
        (fun q => u.readMem p.y (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val)) (length-1-k) := by
  by_cases hz : length-1-k=0
  · simp [RealReference.diagSsmBackwardPrevState,RealReference.reverseTime,Reference.previousState,hz,RealReference.colOffset,column]
  · simp [RealReference.diagSsmBackwardPrevState,RealReference.reverseTime,Reference.previousState,hz,
      Nat.pos_of_ne_zero hz,RealReference.timeOffset,RealReference.colOffset,column]

theorem real_transition_reference (p : BackwardLayout) (u : BlockState) (j : Fin p.B) (length k : Nat) :
    RealReference.diagSsmBackwardGradLambdaAfter u p.s p.l p.y p.gy p.batch p.dim p.B length j k =
      Reference.transitionGradient length (u.readMem p.l (column p.B (u.pids 0) j.val%p.dim))
        (u.readMem p.s (column p.B (u.pids 0) j.val))
        (fun q => u.readMem p.y (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val))
        (fun q => u.readMem p.gy (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val)) k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [RealReference.diagSsmBackwardGradLambdaAfter,Reference.transitionGradient,ih,
      real_carry_reference,real_previous_reference]
    rfl

theorem forward_real_value (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.false p length).Ctx) (hp : (forwardIO Bool.false p length).RegionalInputs u v)
    (t : Fin length) (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    RealReference.diagSsmForwardSpec u p.s p.x p.l p.batch p.dim p.B t.val j =
      forwardFormula p.B (forwardRealInputs p length v) (pair t j).val := by
  have hs := forward_real_pin p length u v hp 0 j.val j.isLt ha
  have hl := forward_real_pin p length u v hp 1 j.val j.isLt ha
  have hx (q : Nat) (hq : q<length) :=
    forward_real_pin p length u v hp 2 (pair (⟨q,hq⟩ : Fin length) j).val
      (pair (⟨q,hq⟩ : Fin length) j).isLt (by simpa [forwardLane] using ha)
  simp only [forwardBuffer,forwardAddress] at hs hl
  simp only [RealReference.diagSsmForwardSpec,real_forward_reference,forwardFormula,pair_div,pair_mod]
  rw [hs,hl]
  apply ReferenceCongruence.forward_prefix
  intro q hq
  have h := hx q (by have := t.isLt; omega)
  dsimp only [forwardBuffer,forwardAddress] at h
  simp only [streamAddress] at h
  simp only [pair_div,pair_mod] at h
  simpa only [pair_val] using h


theorem backward_real_bindings (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.false p length).Ctx) (hp : (backwardIO Bool.false p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    u.readMem p.l (column p.B (u.pids 0) j.val%p.dim)=backwardRealInputs p length v 1 j.val ∧
    (∀ q, q<length → u.readMem p.gy (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val)=
      backwardRealInputs p length v 3 (q*p.B+j.val)) ∧
    (0<length → u.readMem p.s (column p.B (u.pids 0) j.val)=backwardRealInputs p length v 0 j.val) ∧
    (∀ q, q+1<length → u.readMem p.y (q*(p.batch*p.dim)+column p.B (u.pids 0) j.val)=
      backwardRealInputs p length v 2 (q*p.B+j.val)) := by
  refine ⟨backward_real_pin p length u v hp 1 j.val j.isLt (by simp) ha,?_,?_,?_⟩
  · intro q hq
    have h := backward_real_pin p length u v hp 3 (pair (⟨q,hq⟩ : Fin length) j).val
      (pair (⟨q,hq⟩ : Fin length) j).isLt (by simp) (by simpa [backwardLane] using ha)
    dsimp only [backwardBuffer,backwardAddress] at h
    simp only [streamAddress] at h
    simp only [pair_div,pair_mod] at h
    simpa only [pair_val] using h
  · intro hz
    exact backward_real_pin p length u v hp 0 j.val j.isLt (by omega) ha
  · intro q hq
    have hq' : q<length-1 := by omega
    have h := backward_real_pin p length u v hp 2 (pair (⟨q,hq'⟩ : Fin (length-1)) j).val
      (pair (⟨q,hq'⟩ : Fin (length-1)) j).isLt (by simp) (by simpa [backwardLane] using ha)
    dsimp only [backwardBuffer,backwardAddress] at h
    simp only [streamAddress] at h
    simp only [pair_div,pair_mod] at h
    simpa only [pair_val] using h

theorem backward_real_carry (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.false p length).Ctx) (hp : (backwardIO Bool.false p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) (k : Nat) (hk : k≤length) :
    RealReference.diagSsmBackwardGradSAfter u p.l p.gy p.batch p.dim p.B length j k =
      Reference.reverseCarry length (backwardRealInputs p length v 1 j.val)
        (fun q => backwardRealInputs p length v 3 (q*p.B+j.val)) k := by
  obtain ⟨hl,hg,_,_⟩ := backward_real_bindings p length u v hp j ha
  rw [real_carry_reference,hl]
  exact ReferenceCongruence.carry_prefix length _ _ _ k hk hg

theorem backward_real_gx (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.false p length).Ctx) (hp : (backwardIO Bool.false p length).RegionalInputs u v)
    (t : Fin length) (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    RealReference.diagSsmBackwardGradXSpec u p.l p.gy p.batch p.dim p.B length t.val j =
      backwardFormula id p.B length (backwardRealInputs p length v) 0 (pair t j).val := by
  obtain ⟨_,hg,_,_⟩ := backward_real_bindings p length u v hp j ha
  simp only [RealReference.diagSsmBackwardGradXSpec,RealReference.timeOffset,RealReference.colOffset,
    backwardFormula,backwardOutLane,Fin.val_zero,↓reduceIte,pair_mod,pair_div,Reference.inputGradient,id_eq]
  rw [backward_real_carry p length u v hp j ha _ (by omega)]
  exact congrArg (fun z => z+Reference.reverseCarry length (backwardRealInputs p length v 1 j.val)
    (fun q => backwardRealInputs p length v 3 (q*p.B+j.val)) (length-1-t.val)) (hg t.val t.isLt)

theorem backward_real_gl (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.false p length).Ctx) (hp : (backwardIO Bool.false p length).RegionalInputs u v)
    (j : Fin p.B) (ha : active p.batch p.dim p.B (u.pids 0) j.val) :
    RealReference.diagSsmBackwardGradLambdaAfter u p.s p.l p.y p.gy p.batch p.dim p.B length j length =
      backwardFormula id p.B length (backwardRealInputs p length v) 2 j.val := by
  by_cases hz : length=0
  · subst length; rfl
  · obtain ⟨hl,hg,hs,hy⟩ := backward_real_bindings p length u v hp j ha
    rw [real_transition_reference,hl,hs (by omega)]
    exact ReferenceCongruence.transition_prefix length _ _ _ _ _ _ length (le_refl _) hy hg

end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
