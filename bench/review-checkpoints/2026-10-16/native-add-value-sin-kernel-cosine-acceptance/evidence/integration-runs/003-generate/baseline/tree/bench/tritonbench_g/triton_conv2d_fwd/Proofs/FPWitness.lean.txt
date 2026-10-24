import bench.tritonbench_g.triton_conv2d_fwd.FPCorrect
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def one : Word := FP.Value.ofBits .fp32 0x3f800000
def two : Word := FP.Value.ofBits .fp32 0x40000000
def eight : Word := FP.Value.ofBits .fp32 0x41000000
def seven : Word := FP.Value.ofBits .fp32 0x40e00000
def halfway : Word := FP.Value.ofBits .fp32 0x3f801000

def witnessParameters : Parameters where
  Input := "I"
  Weight := "W"
  Output := "O"
  batch_dim := 1
  in_feat_dim := 2
  in_height := 2
  in_width := 2
  out_feat_dim := 1
  out_height := 1
  out_width := 1
  input_batch_stride := 8
  input_in_feat_stride := 4
  input_height_stride := 2
  input_width_stride := 1
  weight_out_feat_stride := 8
  weight_in_feat_stride := 4
  weight_height_stride := 2
  weight_width_stride := 1
  output_batch_stride := 8
  output_out_feat_stride := 4
  output_height_stride := 2
  output_width_stride := 1
  kernel_height := 2
  kernel_width := 2
  stride_height := 1
  stride_width := 1
  padding_height := 0
  padding_width := 0
  groups := 1
  BLOCK_SIZE_BATCH_HEIGHT_WIDTH := 2
  BLOCK_SIZE_IN_FEAT := 1
  BLOCK_SIZE_OUT_FEAT := 2
  tf32 := Bool.true

def witnessState (u : Word) : State Word where
  mem := fun r _ => .mk .fp32 (if r=("I" : RegionName) then halfway else one)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => match d with | .fp32 => u | d => defaultValue d

def lane : CIndex witnessParameters := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)

/-- The rejected global half=id premise is concretely false for this input. -/
theorem half_rounding_occurs :
    halfway.toRat? = some (2049/2048 : Rat) ∧
    QuantizeKvTransform.MixedBits.narrow .ieeeValue halfway = FP.Value.ofBits .fp32 0x3c00 ∧
    QuantizeKvTransform.MixedBits.widen
      (QuantizeKvTransform.MixedBits.narrow .ieeeValue halfway) = one ∧
    halfway≠one := by decide +kernel

/-- Nonempty 2×2×2 nest, with actual narrowing on all eight input operands. -/
theorem multiblock_value :
    bitOutput witnessParameters (witnessState (FP.Value.zero .fp32)) lane=eight := by
  decide +kernel

theorem witness_injective (u : Word) :
    ∀ i j, active witnessParameters (witnessState u) i → active witnessParameters (witnessState u) j →
      outputOffset witnessParameters (witnessState u) i=outputOffset witnessParameters (witnessState u) j → i=j := by
  intro i j hi hj he
  rcases i with ⟨i,k,ui⟩
  rcases j with ⟨j,l,uj⟩
  cases ui; cases uj
  simp [active,witnessParameters,witnessState,batch,bh,bhw,feat,ogd,oh,ow,BHW,OF] at hi hj
  simp [hi.1.1,hi.1.2,hj.1.1,hj.1.2]

theorem multiblock_nonempty (rest : Algebra Word) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDot) (source witnessParameters)
      (witnessState (FP.Value.zero .fp32))=some t ∧
      t.mem "O" 0=.mk .fp32 eight ∧
      t.mem "O" 1=(witnessState (FP.Value.zero .fp32)).mem "O" 1 := by
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.full_software_correctness rest witnessParameters
    (witnessState (FP.Value.zero .fp32)) (witness_injective _)
  have h := hv lane (by decide)
  rw [multiblock_value] at h
  refine ⟨t,he,h,?_⟩
  apply hf
  right
  intro i hi
  rcases i with ⟨i,j,u⟩
  simp [active,witnessParameters,witnessState,batch,bh,bhw,feat,ogd,oh,ow,BHW,OF] at hi
  have hi0 : i.val=0 := by omega
  have hj0 : j.val=0 := by omega
  simp [outputOffset,witnessParameters,witnessState,batch,bh,bhw,feat,ogd,oh,ow,hi0,hj0]

/-- A spatial-padding configuration; other-less load behavior is observable in
an active output, even though all numerical inputs are finite. -/
def paddingParameters : Parameters :=
  {witnessParameters with
    in_feat_dim := 1
    in_height := 1
    in_width := 1
    padding_height := 1
    padding_width := 1}
def paddingLane : CIndex paddingParameters := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)

theorem padding_changes_value :
    bitOutput paddingParameters (witnessState (FP.Value.zero .fp32)) paddingLane=one ∧
    bitOutput paddingParameters (witnessState two) paddingLane=seven ∧ one≠seven := by
  decide +kernel

theorem padding_nonempty (rest : Algebra Word) :
    ∃ t0 t2,
      FP.Structural.exec (backend (software rest) sequentialDot) (source paddingParameters)
        (witnessState (FP.Value.zero .fp32))=some t0 ∧
      FP.Structural.exec (backend (software rest) sequentialDot) (source paddingParameters)
        (witnessState two)=some t2 ∧
      t0.mem "O" 0=.mk .fp32 one ∧ t2.mem "O" 0=.mk .fp32 seven := by
  obtain ⟨t0,he0,hv0,hf0⟩ := FPCorrect.full_software_correctness rest paddingParameters
    (witnessState (FP.Value.zero .fp32)) (witness_injective _)
  obtain ⟨t2,he2,hv2,hf2⟩ := FPCorrect.full_software_correctness rest paddingParameters
    (witnessState two) (witness_injective _)
  have h0 := hv0 paddingLane (by decide)
  have h2 := hv2 paddingLane (by decide)
  rw [padding_changes_value.1] at h0
  rw [padding_changes_value.2.1] at h2
  exact ⟨t0,t2,he0,he2,h0,h2⟩

def infiniteWeightState : State Word :=
  {witnessState (FP.Value.zero .fp32) with
    mem := fun r _ => .mk .fp32 (if r=("W" : RegionName) then FP.Value.infinity .fp32 else one)}

/-- Even the optional clean-undef model must retain 0*Inf in masked source
lanes. A live-only real convolution would incorrectly discard this NaN. -/
theorem padded_zero_times_infinity :
    bitOutput paddingParameters infiniteWeightState paddingLane=FP.Value.qNaN .fp32 := by
  decide +kernel

theorem padded_nan_execution (rest : Algebra Word) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDot) (source paddingParameters)
      infiniteWeightState=some t ∧ t.mem "O" 0=.mk .fp32 (FP.Value.qNaN .fp32) := by
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.full_software_correctness rest paddingParameters
    infiniteWeightState (witness_injective (FP.Value.zero .fp32))
  have h := hv paddingLane (by decide)
  rw [padded_zero_times_infinity] at h
  exact ⟨t,he,h⟩

#axiomsClean half_rounding_occurs
#axiomsClean multiblock_value
#axiomsClean witness_injective
#axiomsClean multiblock_nonempty
#axiomsClean padding_changes_value
#axiomsClean padding_nonempty
#axiomsClean padded_zero_times_infinity
#axiomsClean padded_nan_execution
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
