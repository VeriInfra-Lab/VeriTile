import bench.tritonbench_g.layer_norm_liger.FPCorrect
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum softwareSum)
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 22000

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessSmall : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000
def witnessTwo : FP.Value .fp32 := FP.Value.ofBits .fp32 0x40000000
def witnessHalf : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f000000
def witnessParameters : Parameters where
  y := "y"
  x := "x"
  weight := "w"
  bias := "b"
  mean := "mean"
  rstd := "rstd"
  yStride := 7
  xStride := 7
  weightStride := 19
  biasStride := 23
  meanStride := 1
  rstdStride := 1
  columns := 2
  block := 4
  epsilon := 0

def witnessState : State (FP.Value .fp32) where
  mem := fun r o => .mk .fp32 (if r=("x" : RegionName) ∧ o=1 then witnessSmall else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem literal_two : literal32 (fun _ => witnessTwo) (2 : ℝ)=witnessTwo := by
  norm_num [literal32]

/-- Nonempty four-lane row with two padding lanes. Adding 2^-25 to 1
rounds to 1 before division, so the fp32 mean is 1/2, not the exact mean. -/
theorem witness_mean_rounding :
    bitMean .ieeeValue (fun _ => witnessTwo) witnessParameters witnessState = witnessHalf := by
  unfold bitMean
  change FP.div .ieeeValue _ (literal32 (fun _ => witnessTwo) (2 : ℝ))=witnessHalf
  rw [literal_two]
  decide +kernel

theorem witness_not_exact : witnessSmall.toRat?=some (1/33554432) ∧
    witnessHalf.toRat?=some (1/2) ∧ ((1+1/33554432)/2 : Rat)≠1/2 := by decide +kernel

/-- The two padded lanes contribute (0-1)^2 each. This would be zero
if the source had incorrectly been replaced by a sum over live columns. -/
theorem padded_variance_nonzero :
    bitVariance .ieeeValue (fun _ => witnessTwo) witnessParameters
      {witnessState with mem := fun _ _ => .mk .fp32 witnessOne}=witnessOne := by
  simp only [bitVariance, bitCentered, bitMean, witnessParameters]
  norm_num only [Nat.cast_ofNat]
  simp only [literal_two]
  decide +kernel

/-- Native rsqrt remains a named primitive; the successful source run and
rounded Mean output do not assume any law about it. -/
theorem full_forward_nonempty_rounding (rest : Algebra (FP.Value .fp32))
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    ∃ t, FP.Structural.exec
      (withSum (software rest .ieeeValue (fun _ => witnessTwo) rsqrt32) (softwareSum .ieeeValue))
      (kernel witnessParameters) witnessState=some t ∧
      t.mem "mean" 0=.mk .fp32 witnessHalf ∧
      t.mem "y" 2=witnessState.mem "y" 2 ∧ ((1+1/33554432)/2 : Rat)≠1/2 := by
  obtain ⟨t,he,_,hm,_,hf⟩ := FPCorrect.fp32_scalarops_forward_correctness rest .ieeeValue
    (fun _ => witnessTwo) rsqrt32 witnessParameters witnessState (by decide) (by decide) (by decide)
  refine ⟨t,he,?_,?_,witness_not_exact.2.2⟩
  · simpa only [witness_mean_rounding] using hm
  · apply hf
    simp [Outside, witnessParameters, witnessState, meanOffset, rstdOffset, outputOffset]
    omega

#axiomsClean padded_variance_nonzero
#axiomsClean literal_two
#axiomsClean witness_mean_rounding
#axiomsClean witness_not_exact
#axiomsClean full_forward_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
