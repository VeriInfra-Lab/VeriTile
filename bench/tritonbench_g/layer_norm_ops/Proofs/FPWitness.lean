import bench.tritonbench_g.layer_norm_ops.FPCorrect
import bench.tritonbench_g.layer_norm_ops.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages (literal32)
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def one : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def negativeZero : FP.Value .fp32 := FP.Value.ofBits .fp32 0x80000000
def small : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

def witnessParameters : Parameters where
  x := "x"
  xhat := "xhat"
  weight := "w"
  dy := "dy"
  dx := "dx"
  dw := "dw"
  db := "db"
  mean := "mean"
  rstd := "rstd"
  c1 := "c1"
  c2 := "c2"
  dxBase := "dxBase"
  dres := "dres"
  dresIn := "dresIn"
  xStride := 3
  xhatStride := 3
  dyStride := 3
  dxStride := 3
  dresStride := 3
  dresInStride := 3
  columns := 1
  block := 2

def witnessState : State (FP.Value .fp32) where
  mem := fun r _ => .mk .fp32
    (if r=("dy" : RegionName) then negativeZero else
     if r=("dres" : RegionName) then small else
     if r=("mean" : RegionName) then FP.Value.zero .fp32 else one)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : TileIndex [witnessParameters.block] := (⟨0,by decide⟩,PUnit.unit)

/-- A concrete counterexample to deleting the native initial +0 addition.
The old mathematical helper's raw product/db term is -0, but the Python
accumulator add produces +0 in this IEEE round-to-nearest profile. -/
theorem zero_add_changes_bits :
    FP.mul .ieeeValue negativeZero one=negativeZero ∧
    FP.add .ieeeValue (FP.Value.zero .fp32) (FP.mul .ieeeValue negativeZero one)=FP.Value.zero .fp32 ∧
    FP.add .ieeeValue (FP.Value.zero .fp32) negativeZero=FP.Value.zero .fp32 ∧
    negativeZero≠FP.Value.zero .fp32 := by decide +kernel

theorem witness_row_values (rest : Algebra (FP.Value .fp32)) :
    stored (software rest .ieeeValue (fun _ => FP.Value.zero .fp32))
      (rowDX (software rest .ieeeValue (fun _ => FP.Value.zero .fp32)) witnessParameters witnessState Bool.true lane)=
      .mk .fp32 negativeZero ∧
    stored (software rest .ieeeValue (fun _ => FP.Value.zero .fp32))
      (rowDW (software rest .ieeeValue (fun _ => FP.Value.zero .fp32)) witnessParameters witnessState Bool.true lane)=
      .mk .fp32 (FP.Value.zero .fp32) ∧
    stored (software rest .ieeeValue (fun _ => FP.Value.zero .fp32))
      (rowDB (software rest .ieeeValue (fun _ => FP.Value.zero .fp32)) witnessParameters witnessState lane)=
      .mk .fp32 (FP.Value.zero .fp32) := by
  unfold stored rowDX rowDW rowDB rowC1 rowC2 average wdyValue xhatValue dyValue loaded scalar weight op pad zero
  simp [software, literal32, witnessParameters, witnessState, lane,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  all_goals norm_num
  all_goals decide +kernel

/-- Actual full single-row source execution with all three output bit patterns;
the W tail is undef and all two-lane reduction arithmetic is still evaluated. -/
theorem plain_row_nonempty_signed_zero (rest : Algebra (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (software rest .ieeeValue (fun _ => FP.Value.zero .fp32))
      (plainKernel witnessParameters) witnessState=some t ∧
      t.mem "dx" 0=.mk .fp32 negativeZero ∧
      t.mem "dw" 0=.mk .fp32 (FP.Value.zero .fp32) ∧
      t.mem "db" 0=.mk .fp32 (FP.Value.zero .fp32) ∧
      t.mem "dx" 1=witnessState.mem "dx" 1 := by
  obtain ⟨t,he,_,hv,hf⟩ := FPCorrect.fp32_plain_one_row_correctness
    (software rest .ieeeValue (fun _ => FP.Value.zero .fp32)) witnessParameters witnessState
    (by decide) (by decide) (by decide)
  have h := hv lane (by decide)
  have bits := witness_row_values rest
  refine ⟨t,he,h.1.trans bits.1,h.2.1.trans bits.2.1,h.2.2.trans bits.2.2,?_⟩
  apply hf
  simp [RowOutside, OutsideVector, witnessParameters, witnessState, offset]

theorem residual_rounds (rest : Algebra (FP.Value .fp32)) :
    stored (software rest .ieeeValue (fun _ => FP.Value.zero .fp32))
      (residualValue (software rest .ieeeValue (fun _ => FP.Value.zero .fp32)) witnessParameters witnessState lane)=
      .mk .fp32 one := by
  simp [stored, residualValue, loaded, pad, op, software, literal32,
    witnessParameters, witnessState, lane,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  decide +kernel

theorem residual_nonempty_rounding (rest : Algebra (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (software rest .ieeeValue (fun _ => FP.Value.zero .fp32))
      (residualKernel witnessParameters) witnessState=some t ∧
      t.mem "dx" 0=.mk .fp32 one ∧ t.mem "dresIn" 0=.mk .fp32 one ∧
      (1+1/33554432 : Rat)≠1 := by
  obtain ⟨t,he,hv,_⟩ := FPCorrect.fp32_residual_add_correctness
    (software rest .ieeeValue (fun _ => FP.Value.zero .fp32)) witnessParameters witnessState (by decide)
  have h := hv lane (by decide)
  exact ⟨t,he,h.1.trans (residual_rounds rest),h.2.trans (residual_rounds rest),by decide +kernel⟩

#axiomsClean zero_add_changes_bits
#axiomsClean witness_row_values
#axiomsClean plain_row_nonempty_signed_zero
#axiomsClean residual_rounds
#axiomsClean residual_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
