import bench.tritonbench_g.layer_norm_welfold.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (Post word sequentialFP32 sequentialSum)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 15000

def roundedNormalize [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (s : State (FP.Value .fp32)) (mean rstd X W Bias : RegionName)
    (rnumel RBLOCK : Nat) : TileIndex [RBLOCK] → FP.Value .fp32 := fun i =>
  let active := column s RBLOCK i.1 < rnumel
  let x := word s X (rowAddress s rnumel RBLOCK i) active
  let weight := word s W (column s RBLOCK i.1) active
  let bias := word s Bias (column s RBLOCK i.1) active
  let average := (s.mem mean (s.pids 0)).read .real
  let inverse := (s.mem rstd (s.pids 0)).read .real
  FP.add cfg (FP.mul cfg (FP.mul cfg (FP.sub cfg x average) inverse) weight) bias

specification normalize_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (normalizeChunk mean rstd X W Bias Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (roundedNormalize cfg s mean rstd X W Bias rnumel RBLOCK) := by
  obtain ⟨final, he, hp⟩ := normalize_chunk_fp32 (FP.Software.scalarFP32 rest cfg)
    mean rstd X W Bias Y rnumel RBLOCK s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [normalizeValues, convertedLoad, roundedNormalize, word,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessQuarterUlp : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_rounding :
    FP.add .ieeeValue
      (FP.mul .ieeeValue (FP.mul .ieeeValue
        (FP.sub .ieeeValue witnessOne (FP.Value.zero .fp32)) witnessOne) witnessOne)
      witnessQuarterUlp = witnessOne ∧
    witnessOne.toRat? = some 1 ∧ witnessQuarterUlp.toRat? = some (1 / 33554432) ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

def roundingInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun R _ => .mk .real
    (if R = ("mean" : RegionName) then FP.Value.zero .fp32
     else if R = ("bias" : RegionName) ∨ R = ("out" : RegionName) then witnessQuarterUlp
     else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

specification normalize_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (normalizeChunk "mean" "rstd" "x" "weight" "bias" "out" 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := normalize_software_fp32 rest .ieeeValue
    "mean" "rstd" "x" "weight" "bias" "out" 1 2 roundingInitial
  refine ⟨final, he, ?_, ?_, witness_rounding.2.2.2⟩
  · have hv := hp.1 (⟨0, by decide⟩, PUnit.unit) (by simp [column, roundingInitial])
    exact hv.trans (congrArg (Cell.mk .real) witness_rounding.1)
  · apply hp.2 "out" 1
    right
    intro i hi
    simp [column, roundingInitial] at hi
    simp [rowAddress, column, roundingInitial]
    omega

#axiomsClean normalize_software_fp32
#axiomsClean witness_rounding
#axiomsClean normalize_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
