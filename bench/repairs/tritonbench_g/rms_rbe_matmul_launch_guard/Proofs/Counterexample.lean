import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Contract

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Counterexample
open RmsMatmulRbe.Contract hiding Inputs InputContract checkInputs checkInputs_iff
open Contract

def tensor (dtype : VeriTile.Torch.DType) (shape strides : List Nat) (base bytes : Nat) : Tensor :=
  { tensor := true
    metadata := {
      dtype := dtype
      layout := .strided
      device := ⟨"cuda", 0⟩
      shape := shape
      isContiguous := true
      isNeg := false
      dataPtr := BitVec.ofNat 64 base }
    base := base, bytes := bytes, offset := 0, strides := strides }

noncomputable def witness : Inputs :=
  { x := tensor fp32 [1,1048561,1] [1048561,1,1] 4096 4194244
    weight := tensor fp16 [2,1] [1,1] 8388608 4
    rms := tensor fp16 [1] [1] 16777216 2
    rotaryBool := true, rotary := false
    scalars := {
      startInteger := true
      headsInteger := true
      dimInteger := true
      start := 0
      heads := 1
      dim := 2
      epsilonNumeric := true
      epsilonFinite := true
      thetaNumeric := true
      thetaFinite := true
      epsilon := 1/1000000
      theta := 10000 } }

theorem original_admitted : RmsRbeMatmul.Contract.InputContract witness := by
  norm_num [RmsRbeMatmul.Contract.InputContract, witness, tensor, TensorAdmitted,
    ScalarAdmitted, RmsRbeMatmul.Contract.effectiveScalars, RmsRbeMatmul.Contract.rmsTypes,
    RmsRbeMatmul.Contract.Inputs.reads, SameReadFormat, disjoint, overlap, dim,
    fp32, fp16, int8, weightTypes, width, VeriTile.Torch.TensorMetadata.numel]
  decide

theorem original_checks_pass : RmsRbeMatmul.Contract.checkInputs witness = .ok () :=
  (RmsRbeMatmul.Contract.checkInputs_iff witness).mpr original_admitted

theorem original_grid : grid witness = (1,65536) := by
  norm_num [grid, witness, tensor, dim]

theorem candidate_rejects : ¬ InputContract witness := by
  intro h
  have bounds := nonempty_axes witness h (by norm_num [witness, tensor, dim])
    (by norm_num [witness, tensor, dim])
  rw [original_grid] at bounds
  norm_num at bounds

theorem candidate_checks_reject : checkInputs witness ≠ .ok () := by
  intro passed
  exact candidate_rejects ((Contract.checkInputs_iff witness).mp passed)

#axiomsClean original_admitted
#axiomsClean original_checks_pass
#axiomsClean original_grid
#axiomsClean candidate_rejects
#axiomsClean candidate_checks_reject
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Counterexample
