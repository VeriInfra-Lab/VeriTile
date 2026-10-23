import bench.repairs.tritonbench_g.diag_ssm_triton.IO
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOBackwardResults
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.HeadlineRecurrences

namespace VeriTile.Bench.Repairs.DiagSsmTriton.LogicalIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton Interface
open scoped VeriTile.Triton.LogicalIO
noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option linter.unusedSimpArgs false

/-- Exact finite time/lane extension, including zero-sized dimensions. -/
theorem timeAt_matrix {A : Type} [Zero A] (length B : Nat) (v : Fin (length*B) → A)
    (time : Nat) (lane : Fin B) :
    timeAt (matrix A length B v) time lane =
      (if h : time*B+lane.val<length*B then v ⟨time*B+lane.val,h⟩ else 0) := by
  by_cases ht : time<length
  · have hb : time*B+lane.val<length*B := by
      have h₁ := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ht)
      have h₂ := lane.isLt
      simp only [Nat.succ_mul] at h₁
      omega
    simp only [timeAt,ht,↓reduceDIte,matrix,Equiv.coe_fn_mk,hb]
    congr 1
    apply Fin.ext
    simp [finProdFinEquiv,Nat.mul_comm,Nat.add_comm]
  · have hb : ¬time*B+lane.val<length*B := by
      have h₁ := Nat.mul_le_mul_right B (Nat.le_of_not_gt ht)
      omega
    simp [timeAt,ht,hb]

def forwardMath {A : Type} [Semiring A] {B length : Nat} (xs : ForwardInputs A B length)
    (time : Fin length) (lane : Fin B) : A :=
  Reference.forwardState (xs.initial lane) (xs.transition lane) (fun t => timeAt xs.input t lane) (time.val+1)

def backwardMath {A : Type} [Semiring A] (conjugate : A → A) {B length : Nat}
    (xs : BackwardInputs A B length) : BackwardOutputs A B length :=
  ⟨(fun time lane => Reference.inputGradient length (conjugate (xs.transition lane))
      (fun t => timeAt xs.upstream t lane) time.val),
    (fun lane => Reference.reverseCarry length (conjugate (xs.transition lane))
      (fun t => timeAt xs.upstream t lane) length),
    (fun lane => Reference.transitionGradient length (conjugate (xs.transition lane))
      (conjugate (xs.initial lane)) (fun t => conjugate (timeAt xs.saved t lane))
      (fun t => timeAt xs.upstream t lane) length)⟩

@[simp] theorem forwardReal_initial (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.false p length).Ctx) (lane : Fin p.B) :
    (forwardRealInputs p length v).initial lane = Interface.forwardRealInputs p length v 0 lane.val := by
  change v (0 : Fin 3) lane = _
  simp [Interface.forwardRealInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem forwardReal_transition (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.false p length).Ctx) (lane : Fin p.B) :
    (forwardRealInputs p length v).transition lane = Interface.forwardRealInputs p length v 1 lane.val := by
  change v (1 : Fin 3) lane = _
  simp [Interface.forwardRealInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem forwardReal_input (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.false p length).Ctx) (t : Nat) (lane : Fin p.B) :
    timeAt (forwardRealInputs p length v).input t lane = Interface.forwardRealInputs p length v 2 (t*p.B+lane.val) := by
  change timeAt (matrix ℝ length p.B (v (2 : Fin 3))) t lane = _
  rw [timeAt_matrix]
  rfl

/-- Equality of the complete previous regional judgment and the logical target. -/
theorem forwardReal_iff (p : ForwardLayout) (length : Nat) :
    (forwardRealView p length ⊨ forwardMath) ↔
      (forwardIO Bool.false p length).RegionalImplements (fun _ _ v _ k => forwardFormula (α := ℝ) p.B (Interface.forwardRealInputs p length v) k.val) := by
  have formula : (forwardRealView p length).lift forwardMath =
      (fun v o k => forwardFormula (α := ℝ) p.B (Interface.forwardRealInputs p length v) k.val) := by
    funext v o k
    fin_cases o
    change forwardMath (forwardRealInputs p length v) k.divNat k.modNat = _
    simp only [forwardMath,forwardFormula,forwardReal_initial,forwardReal_transition,forwardReal_input]
    rfl
  change ((forwardIO Bool.false p length).RegionalImplements (fun _ _ v => _) ↔ _)
  rw [formula]
  all_goals rfl

theorem forwardReal_correct (p : ForwardLayout) (length : Nat) (sep : p.x≠p.y) :
    forwardRealView p length ⊨ forwardMath :=
  (forwardReal_iff p length).mpr (forward_real_correct p length sep)

@[simp] theorem forwardComplex_initial (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.true p length).Ctx) (lane : Fin p.B) :
    (forwardComplexInputs p length v).initial lane = Interface.forwardComplexInputs p length v 0 lane.val := by
  change (⟨v (0 : Fin 6) lane,v (1 : Fin 6) lane⟩ : ℂ) = _
  simp [Interface.forwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem forwardComplex_transition (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.true p length).Ctx) (lane : Fin p.B) :
    (forwardComplexInputs p length v).transition lane = Interface.forwardComplexInputs p length v 1 lane.val := by
  change (⟨v (2 : Fin 6) lane,v (3 : Fin 6) lane⟩ : ℂ) = _
  simp [Interface.forwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem forwardComplex_input (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.true p length).Ctx) (t : Nat) (lane : Fin p.B) :
    timeAt (forwardComplexInputs p length v).input t lane = Interface.forwardComplexInputs p length v 2 (t*p.B+lane.val) := by
  change timeAt (matrix ℂ length p.B (fun k => ⟨v (4 : Fin 6) k,v (5 : Fin 6) k⟩)) t lane = _
  rw [timeAt_matrix]
  by_cases h : t*p.B+lane.val<length*p.B
  · simp [Interface.forwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,h] <;> rfl
  · simp [Interface.forwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,h] <;> rfl

/-- Equality of the complete previous regional judgment and the logical target. -/
theorem forwardComplex_iff (p : ForwardLayout) (length : Nat) :
    (forwardComplexView p length ⊨ forwardMath) ↔
      (forwardIO Bool.true p length).RegionalImplements (fun _ _ v o k => if o.val=0 then (forwardFormula (α := ℂ) p.B (Interface.forwardComplexInputs p length v) k.val).re else (forwardFormula (α := ℂ) p.B (Interface.forwardComplexInputs p length v) k.val).im) := by
  have formula : (forwardComplexView p length).lift forwardMath =
      (fun v o k => if o.val=0 then (forwardFormula (α := ℂ) p.B (Interface.forwardComplexInputs p length v) k.val).re else (forwardFormula (α := ℂ) p.B (Interface.forwardComplexInputs p length v) k.val).im) := by
    funext v o k
    fin_cases o
    · change (forwardMath (forwardComplexInputs p length v) k.divNat k.modNat).re = _
      simp only [forwardMath,forwardFormula,forwardComplex_initial,forwardComplex_transition,forwardComplex_input]
      rfl
    · change (forwardMath (forwardComplexInputs p length v) k.divNat k.modNat).im = _
      simp only [forwardMath,forwardFormula,forwardComplex_initial,forwardComplex_transition,forwardComplex_input]
      rfl
  change ((forwardIO Bool.true p length).RegionalImplements (fun _ _ v => _) ↔ _)
  rw [formula]
  all_goals rfl

theorem forwardComplex_correct (p : ForwardLayout) (length : Nat) :
    forwardComplexView p length ⊨ forwardMath :=
  (forwardComplex_iff p length).mpr (forward_complex_correct p length)

@[simp] theorem backwardReal_initial (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.false p length).Ctx) (lane : Fin p.B) :
    (backwardRealInputs p length v).initial lane = Interface.backwardRealInputs p length v 0 lane.val := by
  change v (0 : Fin 4) lane = _
  simp [Interface.backwardRealInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem backwardReal_transition (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.false p length).Ctx) (lane : Fin p.B) :
    (backwardRealInputs p length v).transition lane = Interface.backwardRealInputs p length v 1 lane.val := by
  change v (1 : Fin 4) lane = _
  simp [Interface.backwardRealInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem backwardReal_saved (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.false p length).Ctx) (t : Nat) (lane : Fin p.B) :
    timeAt (backwardRealInputs p length v).saved t lane = Interface.backwardRealInputs p length v 2 (t*p.B+lane.val) := by
  change timeAt (matrix ℝ (length-1) p.B (v (2 : Fin 4))) t lane = _
  rw [timeAt_matrix]
  rfl

@[simp] theorem backwardReal_upstream (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.false p length).Ctx) (t : Nat) (lane : Fin p.B) :
    timeAt (backwardRealInputs p length v).upstream t lane = Interface.backwardRealInputs p length v 3 (t*p.B+lane.val) := by
  change timeAt (matrix ℝ length p.B (v (3 : Fin 4))) t lane = _
  rw [timeAt_matrix]
  rfl

/-- Equality of the complete previous regional judgment and the logical target. -/
theorem backwardReal_iff (p : BackwardLayout) (length : Nat) :
    (backwardRealView p length ⊨ backwardMath id) ↔
      (backwardIO Bool.false p length).RegionalImplements (fun _ _ v o k => backwardFormula (α := ℝ) id p.B length (Interface.backwardRealInputs p length v) o k.val) := by
  have formula : (backwardRealView p length).lift (backwardMath id) =
      (fun v o k => backwardFormula (α := ℝ) id p.B length (Interface.backwardRealInputs p length v) o k.val) := by
    funext v
    change (backwardRealOutputs p length).symm (backwardMath id (backwardRealInputs p length v)) = _
    apply (backwardRealOutputs p length).injective
    rw [Equiv.apply_symm_apply]
    apply BackwardOutputs.ext
    · funext time lane
      change Reference.inputGradient length ((backwardRealInputs p length v).transition lane)
        (fun t => timeAt (backwardRealInputs p length v).upstream t lane) time.val = _
      simp only [backwardReal_transition,backwardReal_upstream]
      simp [backwardRealOutputs,matrix,Equiv.coe_fn_mk,backwardFormula,backwardOutLane,
        finProdFinEquiv,Interface.backwardRealInputs,extend,backwardIO,backwardArity,components,
        Nat.mod_eq_of_lt lane.isLt,Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt lane.isLt),
        Nat.div_eq_of_lt lane.isLt,Nat.add_mod,Nat.mul_mod]
    · funext lane
      change Reference.reverseCarry length ((backwardRealInputs p length v).transition lane)
        (fun t => timeAt (backwardRealInputs p length v).upstream t lane) length = _
      simp only [backwardReal_transition,backwardReal_upstream]
      rfl
    · funext lane
      change Reference.transitionGradient length ((backwardRealInputs p length v).transition lane)
        ((backwardRealInputs p length v).initial lane)
        (fun t => timeAt (backwardRealInputs p length v).saved t lane)
        (fun t => timeAt (backwardRealInputs p length v).upstream t lane) length = _
      simp only [backwardReal_transition,backwardReal_initial,backwardReal_saved,backwardReal_upstream]
      rfl
  change ((backwardIO Bool.false p length).RegionalImplements (fun _ _ v => _) ↔ _)
  rw [formula]
  all_goals rfl

theorem backwardReal_correct (p : BackwardLayout) (length : Nat) (sep : RegionCorrectness.BackwardSeparation p) :
    backwardRealView p length ⊨ backwardMath id :=
  (backwardReal_iff p length).mpr (backward_real_correct p length sep)

@[simp] theorem backwardComplex_initial (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.true p length).Ctx) (lane : Fin p.B) :
    (backwardComplexInputs p length v).initial lane = Interface.backwardComplexInputs p length v 0 lane.val := by
  change (⟨v (0 : Fin 8) lane,v (1 : Fin 8) lane⟩ : ℂ) = _
  simp [Interface.backwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem backwardComplex_transition (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.true p length).Ctx) (lane : Fin p.B) :
    (backwardComplexInputs p length v).transition lane = Interface.backwardComplexInputs p length v 1 lane.val := by
  change (⟨v (2 : Fin 8) lane,v (3 : Fin 8) lane⟩ : ℂ) = _
  simp [Interface.backwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,lane.isLt]

@[simp] theorem backwardComplex_saved (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.true p length).Ctx) (t : Nat) (lane : Fin p.B) :
    timeAt (backwardComplexInputs p length v).saved t lane = Interface.backwardComplexInputs p length v 2 (t*p.B+lane.val) := by
  change timeAt (matrix ℂ (length-1) p.B (fun k => ⟨v (4 : Fin 8) k,v (5 : Fin 8) k⟩)) t lane = _
  rw [timeAt_matrix]
  by_cases h : t*p.B+lane.val<(length-1)*p.B
  · simp [Interface.backwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,h] <;> rfl
  · simp [Interface.backwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,h] <;> rfl
  all_goals rfl

@[simp] theorem backwardComplex_upstream (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.true p length).Ctx) (t : Nat) (lane : Fin p.B) :
    timeAt (backwardComplexInputs p length v).upstream t lane = Interface.backwardComplexInputs p length v 3 (t*p.B+lane.val) := by
  change timeAt (matrix ℂ length p.B (fun k => ⟨v (6 : Fin 8) k,v (7 : Fin 8) k⟩)) t lane = _
  rw [timeAt_matrix]
  by_cases h : t*p.B+lane.val<length*p.B
  · simp [Interface.backwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,h] <;> rfl
  · simp [Interface.backwardComplexInputs,extend,forwardIO,backwardIO,forwardArity,backwardArity,components,h] <;> rfl

/-- Equality of the complete previous regional judgment and the logical target. -/
theorem backwardComplex_iff (p : BackwardLayout) (length : Nat) :
    (backwardComplexView p length ⊨ backwardMath (starRingEnd ℂ)) ↔
      (backwardIO Bool.true p length).RegionalImplements (fun _ _ v o k => let z := backwardFormula (α := ℂ) (starRingEnd ℂ) p.B length (Interface.backwardComplexInputs p length v) o.divNat k.val; if o.modNat.val=0 then z.re else z.im) := by
  have formula : (backwardComplexView p length).lift (backwardMath (starRingEnd ℂ)) =
      (fun v o k => let z := backwardFormula (α := ℂ) (starRingEnd ℂ) p.B length (Interface.backwardComplexInputs p length v) o.divNat k.val; if o.modNat.val=0 then z.re else z.im) := by
    funext v
    change (backwardComplexOutputs p length).symm
      (backwardMath (starRingEnd ℂ) (backwardComplexInputs p length v)) = _
    apply (backwardComplexOutputs p length).injective
    rw [Equiv.apply_symm_apply]
    apply BackwardOutputs.ext
    · funext time lane
      have positive : 0<p.B := Nat.zero_lt_of_lt lane.isLt
      simp only [backwardComplexOutputs,Equiv.coe_fn_mk,matrix,backwardMath]
      simp [backwardFormula,backwardOutLane,Fin.divNat,Fin.modNat,finProdFinEquiv,
        backwardComplex_initial,backwardComplex_transition,backwardComplex_saved,backwardComplex_upstream,
        components,Nat.mod_eq_of_lt lane.isLt,Nat.div_eq_of_lt lane.isLt,
        Nat.not_le_of_gt lane.isLt,Nat.add_div,Nat.add_mod,positive]
    · funext lane
      simp only [backwardComplexOutputs,Equiv.coe_fn_mk,backwardMath]
      simp [backwardFormula,backwardOutLane,Fin.divNat,Fin.modNat,
        backwardComplex_initial,backwardComplex_transition,backwardComplex_saved,backwardComplex_upstream,components]
    · funext lane
      simp only [backwardComplexOutputs,Equiv.coe_fn_mk,backwardMath]
      simp [backwardFormula,backwardOutLane,Fin.divNat,Fin.modNat,
        backwardComplex_initial,backwardComplex_transition,backwardComplex_saved,backwardComplex_upstream,components]
  change ((backwardIO Bool.true p length).RegionalImplements (fun _ _ v => _) ↔ _)
  rw [formula]
  all_goals rfl

theorem backwardComplex_correct (p : BackwardLayout) (length : Nat) (sep : ComplexBackwardRun.Separation p) :
    backwardComplexView p length ⊨ backwardMath (starRingEnd ℂ) :=
  (backwardComplex_iff p length).mpr (backward_complex_correct p length sep)

end
end VeriTile.Bench.Repairs.DiagSsmTriton.LogicalIO
