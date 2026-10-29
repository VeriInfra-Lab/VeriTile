import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPCharacterizationCase
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def separateWeightBits (mode : Bool) (step : Nat) : Nat := if mode || step=0 then 0x3c02 else 0x3c00
def fusedWeightBits (mode : Bool) (step : Nat) : Nat := if mode || step=0 then 0x3c01 else 0x3c00

theorem separate_weight_uniform (mode : Bool) (step : Nat) (hs : step<2)
    (i : BIndex (parameters mode)) :
    FPSoftware.dequantized (parameters mode) initial step i = halfWord (separateWeightBits mode step) := by
  simp only [FPSoftware.dequantized,FPSoftware.scaledZero,weight_uniform,zero_uniform,scale_uniform mode step hs]
  cases mode <;> interval_cases step <;> decide +kernel

theorem fused_weight_uniform (mode : Bool) (step : Nat) (hs : step<2)
    (i : BIndex (parameters mode)) :
    FPFused.fusedWeight (parameters mode) initial step i = halfWord (fusedWeightBits mode step) := by
  simp only [FPFused.fusedWeight,weight_uniform,zero_uniform,scale_uniform mode step hs]
  cases mode <;> interval_cases step <;> decide +kernel

def separateOutputBits (mode : Bool) : Nat := if mode then 0x5402 else 0x5401
def fusedOutputBits (mode : Bool) : Nat := if mode then 0x5401 else 0x5400

theorem separate_output_all_lanes (mode : Bool) (i : CIndex (parameters mode)) :
    FPCompiled.plannedOutput FPCompiled.ascendingPlan (parameters mode) initial i = halfWord (separateOutputBits mode) := by
  unfold FPCompiled.plannedOutput
  rw [show steps (parameters mode)=2 by norm_num [steps,parameters]]
  change QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (FPCompiled.plannedAccum FPCompiled.ascendingPlan (parameters mode) initial 2 i) = _
  simp only [FPCompiled.plannedAccum,a_uniform,
    separate_weight_uniform mode 0 (by decide),separate_weight_uniform mode 1 (by decide)]
  cases mode <;> decide +kernel

theorem fused_output_all_lanes (mode : Bool) (i : CIndex (parameters mode)) :
    FPFused.plannedOutput FPCompiled.ascendingPlan (parameters mode) initial i = halfWord (fusedOutputBits mode) := by
  unfold FPFused.plannedOutput
  rw [show steps (parameters mode)=2 by norm_num [steps,parameters]]
  change QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (FPFused.plannedAccum FPCompiled.ascendingPlan (parameters mode) initial 2 i) = _
  simp only [FPFused.plannedAccum,a_uniform,
    fused_weight_uniform mode 0 (by decide),fused_weight_uniform mode 1 (by decide)]
  cases mode <;> decide +kernel

theorem original_output_all_lanes (mode : Bool) (i : CIndex (parameters mode)) :
    SourceBits.sourceOutput (parameters mode) initial i = halfWord (separateOutputBits mode) := by
  unfold SourceBits.sourceOutput
  rw [show steps (parameters mode)=2 by norm_num [steps,parameters]]
  change QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (SourceBits.sourceAccum (parameters mode) initial 2 i) = _
  simp only [SourceBits.sourceAccum,SourceBits.blockBits,a_uniform,
    separate_weight_uniform mode 0 (by decide),separate_weight_uniform mode 1 (by decide)]
  cases mode <;> decide +kernel

theorem distinguish_fusion_both_groups (mode : Bool) :
    halfWord (separateOutputBits mode) ≠ halfWord (fusedOutputBits mode) := by
  cases mode <;> decide +kernel

#axiomsClean separate_weight_uniform
#axiomsClean fused_weight_uniform
#axiomsClean separate_output_all_lanes
#axiomsClean fused_output_all_lanes
#axiomsClean original_output_all_lanes
#axiomsClean distinguish_fusion_both_groups
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
