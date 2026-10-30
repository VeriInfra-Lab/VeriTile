import bench.tritonbench_g.int8_quantization.Proofs.SourceReal
namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Math
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 600000

theorem bad_nonzero (rest : Algebra ℝ) (p : Parameters) (positive : 0 < p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (invalid : ¬FiniteDomain p entry) :
    badValue (algebra rest) primitives p positive entry ≠ 0 := by
  unfold FiniteDomain at invalid
  push Not at invalid
  obtain ⟨i,active,large⟩:=invalid
  have flag : flags (algebra rest) primitives p entry i=Bool.false := by
    unfold flags
    rw [transformed_ideal rest p entry factor]
    norm_num only [algebra,primitives,ideal,if_pos active,decide_eq_false_iff_not]
    exact not_le.mpr large
  have below := Finset.le_sup' (f:=fun j : Fin p.width=>
    if flags (algebra rest) primitives p entry (j,PUnit.unit) then (0:Int) else 1) (Finset.mem_univ i.1)
  have lane : (i.1,PUnit.unit)=i := by cases i; rfl
  rw [lane,flag] at below
  change 1 ≤ badValue (algebra rest) primitives p positive entry at below
  omega

theorem quantized_invalid (rest : Algebra ℝ) (p : Parameters) (positive : 0 < p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (invalid : ¬FiniteDomain p entry)
    (i : Lane p) : quantized (algebra rest) primitives p positive entry i=0 := by
  have bad:=bad_nonzero rest p positive entry factor invalid
  simp only [quantized,biased,clipped,normalized,usable,bad,decide_false,Bool.false_and,
    Bool.false_eq_true,↓reduceIte]
  norm_num [algebra,primitives,scalar32,
    tritonTruncTowardZero]

theorem scale_invalid (rest : Algebra ℝ) (p : Parameters) (positive : 0 < p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (invalid : ¬FiniteDomain p entry) :
    savedScale (algebra rest) primitives p positive entry=0 := by
  rw [savedScale,if_neg (bad_nonzero rest p positive entry factor invalid)]
  norm_num [scalar32,algebra]

#axiomsClean bad_nonzero
#axiomsClean quantized_invalid
#axiomsClean scale_invalid
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Math
