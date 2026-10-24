import bench.tritonbench_g.rowwise_quantization_triton.FPCorrect

namespace VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.TypedScaled
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- A mixed-NaN active row and a separate inactive lane. The old public
hypotheses admit this input: there is no finite/non-NaN restriction. -/
def maxnumParameters : Parameters where
  x := "x"
  out := "out"
  maxOut := "max"
  columns := 2
  width := 3

def maxnumState : State Word where
  mem := fun r o =>
    if r = "x" then .mk .real (if o = 0 then FP.Value.ofBits .fp32 0x3f800000
      else FP.Value.qNaN .fp32)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => match dtype with
    | .real => FP.Value.qNaN .fp32
    | other => defaultValue other

theorem maxnum_mixed_nan_values :
    bitMaximum .ieeeValue maxnumParameters maxnumState = FP.Value.ofBits .fp32 0x3f800000 ∧
    bitValue .ieeeValue maxnumParameters maxnumState (⟨0, by decide⟩, PUnit.unit) =
      FP.Value.ofBits .fp32 0x42fe0000 ∧
    bitValue .ieeeValue maxnumParameters maxnumState (⟨1, by decide⟩, PUnit.unit) =
      FP.Value.qNaN .fp32 ∧
    bitMaximumStored .ieeeValue maxnumParameters maxnumState = FP.Value.ofBits .fp32 0x3c00 ∧
    (QuantizeKvCopy.TypedQuantize.maximum (fun i : Fin 3 =>
      if i.val < 2 then ((maxnumState.mem "x" i.val).read .real).abs
      else FP.Value.zero .fp32)).isNaN = Bool.true := by
  decide +kernel

/-- Actual source execution distinguishes the repaired maxnum profile from
its old rank-based instance: the numeric output is 127 and the stored maximum
is half 1, while the active NaN remains NaN and the inactive output is untouched.
This supporting fixed case does not replace the parameterized public contract. -/
theorem maxnum_mixed_nan_execution (rest : Algebra Word) :
    ∃ t, FP.Structural.exec (bitBackend rest .ieeeValue) (kernel maxnumParameters)
        maxnumState = some t ∧
      t.mem "out" 0 = .mk .real (FP.Value.ofBits .fp32 0x42fe0000) ∧
      t.mem "out" 1 = .mk .real (FP.Value.qNaN .fp32) ∧
      t.mem "max" 0 = .mk .fp16 (FP.Value.ofBits .fp32 0x3c00) ∧
      t.mem "out" 2 = .mk .int 77 := by
  obtain ⟨t, he, hv, hm, hf⟩ := FPCorrect.rowwise_scaled_software_correctness
    rest .ieeeValue maxnumParameters maxnumState (by decide) (by decide)
  refine ⟨t, he, ?_, ?_, ?_, ?_⟩
  · have h := hv (⟨0, by decide⟩, PUnit.unit) (by decide)
    rw [maxnum_mixed_nan_values.2.1] at h
    simpa only [offset, maxnumParameters, maxnumState, Nat.zero_mul, Nat.zero_add] using h
  · have h := hv (⟨1, by decide⟩, PUnit.unit) (by decide)
    rw [maxnum_mixed_nan_values.2.2.1] at h
    simpa only [offset, maxnumParameters, maxnumState, Nat.zero_mul, Nat.zero_add] using h
  · rw [maxnum_mixed_nan_values.2.2.2.1] at hm
    exact hm
  · rw [hf "out" 2 ?_]
    · rfl
    · constructor
      · exact Or.inl (by decide)
      · right
        intro i hi
        simp only [maxnumParameters] at hi
        simp only [offset, maxnumParameters, maxnumState, Nat.zero_mul, Nat.zero_add]
        omega

#axiomsClean maxnum_mixed_nan_values
#axiomsClean maxnum_mixed_nan_execution
end
end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.TypedScaled
