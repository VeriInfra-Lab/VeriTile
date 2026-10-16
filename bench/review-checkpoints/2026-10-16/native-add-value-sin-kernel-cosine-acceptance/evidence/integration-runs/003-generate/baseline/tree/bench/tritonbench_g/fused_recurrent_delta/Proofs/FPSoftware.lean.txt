import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPState
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

/-- A fully specified software reduction: ascending ax index, left fold,
positive zero initial accumulator, and one rounded binary32 addition per lane.
It is an explicit software profile, not a claim about a CUDA reduction tree. -/
def sequentialSum (cfg : FP.Config) {shape : TileShape} (ax : Fin shape.length)
    (keepDims : Bool) (x : Values (FP.Value .fp32) .real shape) :
    Values (FP.Value .fp32) .real (TileShape.reduceShape shape ax keepDims) := by
  cases keepDims
  · exact fun out => (List.finRange (TileShape.axisDim shape ax)).foldl
      (fun a k => FP.add cfg a (x (TileShape.insertAxisIndex shape ax out k))) (FP.Value.zero .fp32)
  · exact fun out => (List.finRange (TileShape.axisDim shape ax)).foldl
      (fun a k => FP.add cfg a (x (TileShape.replaceAxisIndex shape ax out k))) (FP.Value.zero .fp32)

def sequentialFP32 (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  { FP.Software.scalarFP32 rest cfg with
    reduceSum := fun p _ ax keepDims xs =>
      if FP.Software.fp32Context p .real then sequentialSum cfg ax keepDims xs
      else rest.reduceSum p ax keepDims xs }

def word [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32)) (R : RegionName)
    (off : Nat) (active : Prop) [Decidable active] : FP.Value .fp32 :=
  if active then (s.mem R off).read .real else FP.Value.zero .fp32

def roundedReadout [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (HPrev k : RegionName) (t stride K V BK BV : Nat) : TileIndex [BV] → FP.Value .fp32 :=
  sequentialSum cfg (shape := [BV,BK]) ⟨1, by simp⟩ Bool.false (fun i =>
    FP.mul cfg
      (word s HPrev (stateAddress s K V BK BV i) (stateActive s K V BK BV i))
      (word s k (keyAddress s t stride K BK i.2) (keyIndex s BK i.2.1 < K)))

def roundedResidual [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) : TileIndex [BV] → FP.Value .fp32 := fun i =>
  FP.sub cfg (word s v (valueAddress s t s_vo_h V BV i) (valueIndex s BV i.1 < V))
    (roundedReadout cfg s HPrev k t s_qk_h K V BK BV i)

specification vnew_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HPrev k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      Post s final v BV (valueAddress s t s_vo_h V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedResidual cfg s HPrev k v t s_qk_h s_vo_h K V BK BV) := by
  obtain ⟨final, he, hp⟩ := vnew_fp32_stage (sequentialFP32 rest cfg) HPrev k v t s_qk_h s_vo_h K V BK BV s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [vnewValues, residualValues, readoutValues, roundedResidual, roundedReadout,
    sequentialFP32, maskedInput, word, stateActive, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def roundedOutputOne [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (HNext q : RegionName) (t stride K V BK BV : Nat) : TileIndex [BV] → FP.Value .fp32 :=
  sequentialSum cfg (shape := [BV,BK]) ⟨1, by simp⟩ Bool.false (fun i =>
    FP.mul cfg
      (word s HNext (stateAddress s K V BK BV i) (stateActive s K V BK BV i))
      (FP.mul cfg (word s q (keyAddress s t stride K BK i.2) (keyIndex s BK i.2.1 < K))
        (FP.Value.ofBits .fp32 0x3f800000)))

specification output_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HNext q o : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (output HNext q o t s_qk_h s_vo_h B H K V BK BV 1) s = some final ∧
      Post s final o BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedOutputOne cfg s HNext q t s_qk_h K V BK BV) := by
  obtain ⟨final, he, hp⟩ := output_fp32_stage (sequentialFP32 rest cfg) HNext q o t s_qk_h s_vo_h B H K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [outputValues, roundedOutputOne, sequentialFP32, maskedInput, word, stateActive,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def roundedState [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (HPrev k v beta : RegionName) (headwise : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    TileIndex [BV,BK] → FP.Value .fp32 := fun i =>
  let previous := word s HPrev (stateAddress s K V BK BV i) (stateActive s K V BK BV i)
  let key := word s k (keyAddress s t s_qk_h K BK i.2) (keyIndex s BK i.2.1 < K)
  let b := if headwise then word s beta (valueAddress s t s_vo_h V BV (i.1,PUnit.unit)) (valueIndex s BV i.1 < V)
    else (s.mem beta (s.pids 2 * T + t)).read .real
  let residual := roundedResidual cfg s HPrev k v t s_qk_h s_vo_h K V BK BV (i.1,PUnit.unit)
  FP.add cfg previous (FP.mul cfg key (FP.mul cfg residual b))

specification state_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HPrev k v beta HOut : RegionName)
    (headwise : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (stateStep HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV) s = some final ∧
      final.mem = (stateScatter s HOut K V BK BV
        (roundedState cfg s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV)).mem := by
  obtain ⟨final, he, hm⟩ := state_fp32_stage (sequentialFP32 rest cfg) HPrev k v beta HOut headwise
    t s_qk_h s_vo_h T K V BK BV s
  refine ⟨final, he, ?_⟩
  have hv : stateValues (sequentialFP32 rest cfg) s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV =
      roundedState cfg s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV := by
    funext i
    simp [stateValues, residualValues, readoutValues, roundedState, roundedResidual, roundedReadout,
      sequentialFP32, maskedInput, word, stateActive, FP.Software.scalarFP32, FP.Software.fp32Context]
  simpa only [hv] using hm

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessQuarterUlp : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_reduction_rounding :
    FP.add .ieeeValue
      (FP.add .ieeeValue (FP.Value.zero .fp32)
        (FP.mul .ieeeValue witnessOne (FP.mul .ieeeValue witnessOne witnessOne)))
      (FP.mul .ieeeValue witnessQuarterUlp (FP.mul .ieeeValue witnessOne witnessOne)) = witnessOne ∧
    witnessOne.toRat? = some 1 ∧ witnessQuarterUlp.toRat? = some (1 / 33554432) ∧
    (1 + 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

def roundingInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun R off => .mk .real
    (if R = ("q" : RegionName) ∨ (R = ("state" : RegionName) ∧ off = 0)
      then witnessOne else witnessQuarterUlp)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Two key terms undergo a real software reduction, with one valid value
lane and one masked tail. The low term is lost to nearest-even rounding. -/
specification output_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest .ieeeValue)
      (output "state" "q" "out" 0 2 1 1 1 2 1 2 2 1) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := output_software_fp32 rest .ieeeValue
    "state" "q" "out" 0 2 1 1 1 2 1 2 2 roundingInitial
  refine ⟨final, he, ?_, ?_, witness_reduction_rounding.2.2.2⟩
  · have hv := hp.1 (⟨0, by decide⟩, PUnit.unit) (by simp [valueIndex, roundingInitial])
    exact hv.trans (congrArg (Cell.mk .real) witness_reduction_rounding.1)
  · apply hp.2 "out" 1
    right
    intro i hi
    simp [valueIndex, roundingInitial] at hi
    simp [outputAddress, roundingInitial]
    omega

#axiomsClean witness_reduction_rounding
#axiomsClean output_nonempty_rounding

#axiomsClean vnew_software_fp32
#axiomsClean output_software_fp32
#axiomsClean state_software_fp32
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
