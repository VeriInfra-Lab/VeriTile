import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeContract
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
local instance witnessWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def witnessParameters : Parameters where
  input := "input"
  output := "quantized"
  scales := "scales"
  rows := 3
  columns := 3
  inputRowStride := 10
  inputColumnStride := 1
  outputRowStride := 100
  outputColumnStride := 1
  scaleStride := 2
  blockRows := 1
  blockColumns := 2
  steps := 3

private theorem rowZero (r : Fin witnessParameters.blockRows) : r=⟨0,by decide⟩ := by
  apply Fin.ext
  have hr := r.isLt
  change r.val<1 at hr
  change r.val=0
  omega

def witnessState : State Word where
  mem := fun r o => if r="input" then .mk .real (FP.Value.ofBits .fp32
      (if o=10 then 0x42fe0000 else if o=11 then 0xc27e0000 else if o=12 then 0x41fe0000 else 0xffa12345))
    else if r="scales" then .mk .fp16 (FP.Value.ofBits .fp32 0x1234) else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 3
  undef := fun dtype _ _ => defaultValue dtype

theorem witnessDefined : InputDefined .ieeeValue witnessParameters witnessState := by
  intro k hk i hi
  change k<3 at hk
  interval_cases k
  all_goals rcases i with ⟨r,j,tail⟩
  all_goals cases tail
  all_goals have hr : r=⟨0,by decide⟩ := rowZero _
  all_goals subst r
  all_goals fin_cases j
  all_goals first
  | have : False := by simp [active,witnessParameters] at hi
    exact this.elim
  | decide +kernel

theorem witnessInjective : ∀ a b : WriteIndex witnessParameters,
    a.1<witnessParameters.steps → b.1<witnessParameters.steps →
    active witnessParameters a.1 a.2 → active witnessParameters b.1 b.2 →
    outputOffset witnessParameters witnessState a.1 a.2 = outputOffset witnessParameters witnessState b.1 b.2 → a=b := by
  intro a b ha hb haa hba he
  rcases a with ⟨ka,ra,ja,ua⟩
  rcases b with ⟨kb,rb,jb,ub⟩
  cases ua
  cases ub
  have hra : ra=⟨0,by decide⟩ := rowZero _
  have hrb : rb=⟨0,by decide⟩ := rowZero _
  subst ra
  subst rb
  have hja : ja.val<2 := ja.isLt
  have hjb : jb.val<2 := jb.isLt
  simp [outputOffset,row,witnessParameters,witnessState] at he
  have heq : ka=kb := by omega
  have hj : ja.val=jb.val := by omega
  exact Prod.ext heq (Prod.ext rfl (Prod.ext (Fin.ext hj) rfl))

private theorem witnessValues :
    globalExpected .ieeeValue witnessParameters witnessState ⟨0,by decide⟩ 0=127 ∧
    globalExpected .ieeeValue witnessParameters witnessState ⟨0,by decide⟩ 1=(-63) ∧
    globalExpected .ieeeValue witnessParameters witnessState ⟨0,by decide⟩ 2=31 ∧
    QuantizeKvTransform.MixedBits.narrow .ieeeValue
      (scale .ieeeValue witnessParameters witnessState ⟨0,by decide⟩)=FP.Value.ofBits .fp32 0x3c00 := by
  decide +kernel

/-- Three K iterations over a ragged three-element row: one full block,
one partial block, and one entirely masked block. Both passes execute. -/
theorem bitSuccess (rest : Algebra Word) (u : Nat → Word → Int) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue u) (source witnessParameters) witnessState=some final ∧
      final.mem "quantized" 100=.mk .int 127 ∧ final.mem "quantized" 101=.mk .int (-63) ∧
      final.mem "quantized" 102=.mk .int 31 ∧
      final.mem "scales" 2=.mk .fp16 (FP.Value.ofBits .fp32 0x3c00) ∧
      final.mem "quantized" 103=.mk .int 77 := by
  obtain ⟨final,he,h⟩ := correctness rest .ieeeValue u witnessParameters witnessState
    (by decide) (by decide) (by decide) (by decide) witnessDefined witnessInjective
  refine ⟨final,he,?_,?_,?_,?_,?_⟩
  · have hv := (h.1 ⟨0,by decide⟩ ⟨0,by decide⟩).1
    rw [witnessValues.1] at hv
    simpa [row,witnessParameters,witnessState] using hv
  · have hv := (h.1 ⟨0,by decide⟩ ⟨1,by decide⟩).1
    rw [witnessValues.2.1] at hv
    simpa [row,witnessParameters,witnessState] using hv
  · have hv := (h.1 ⟨0,by decide⟩ ⟨2,by decide⟩).1
    rw [witnessValues.2.2.1] at hv
    simpa [row,witnessParameters,witnessState] using hv
  · have hs := h.2.1 ⟨0,by decide⟩
    rw [witnessValues.2.2.2] at hs
    simpa [scaleOffset,witnessParameters,witnessState] using hs
  · have hf := h.2.2 "quantized" 103 (Or.inr (by
      intro k hk i hi
      have hj : i.2.1.val<2 := i.2.1.isLt
      have hr : i.1=⟨0,by decide⟩ := rowZero _
      simp [active,witnessParameters] at hi
      simp [outputOffset,row,witnessParameters,witnessState,hr]
      omega)) (Or.inl (by decide))
    simpa [witnessState] using hf

#axiomsClean witnessDefined
#axiomsClean witnessInjective
#axiomsClean bitSuccess
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
