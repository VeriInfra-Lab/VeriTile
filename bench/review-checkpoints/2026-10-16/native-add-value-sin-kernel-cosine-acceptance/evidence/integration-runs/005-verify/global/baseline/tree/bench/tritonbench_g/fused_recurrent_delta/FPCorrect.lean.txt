import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPSoftware
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPShape
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPStateShape
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardShape
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardSoftware

/-! Public finite-precision contracts for the previously selected forward
and backward stage slices. Complete recurrent/reverse loops and host
integration are not proved by these statements. The bit-level instance uses an
explicit sequential reduction profile, without asserting native lowering. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPCorrect
open VeriTile.Triton FP.Structural FPStages
noncomputable section
specification output_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (output HNext q o t s_qk_h s_vo_h B H K V BK BV scale) s = some final ∧
      Post s final o BV (outputAddress s t s_vo_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s HNext q t s_qk_h K V BK BV scale) := by
  exact FPStages.output_fp32_stage M HNext q o t s_qk_h s_vo_h B H K V BK BV scale s

specification vnew_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      Post s final v BV (valueAddress s t s_vo_h V BV)
        (fun i => valueIndex s BV i.1 < V) (vnewValues M s HPrev k v t s_qk_h s_vo_h K V BK BV) := by
  exact FPStages.vnew_fp32_stage M HPrev k v t s_qk_h s_vo_h K V BK BV s

specification state_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta HOut : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M
      (stateStep HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV) s = some final ∧
      final.mem = (stateScatter s HOut K V BK BV
        (stateValues M s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV)).mem := by
  exact FPStages.state_fp32_stage M HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV s

specification vnew_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HPrev k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      Post s final v BV (valueAddress s t s_vo_h V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedResidual cfg s HPrev k v t s_qk_h s_vo_h K V BK BV) := by
  exact FPStages.vnew_software_fp32 rest cfg HPrev k v t s_qk_h s_vo_h K V BK BV s

specification output_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HNext q o : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (output HNext q o t s_qk_h s_vo_h B H K V BK BV 1) s = some final ∧
      Post s final o BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedOutputOne cfg s HNext q t s_qk_h K V BK BV) := by
  exact FPStages.output_software_fp32 rest cfg HNext q o t s_qk_h s_vo_h B H K V BK BV s

specification state_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HPrev k v beta HOut : RegionName)
    (headwise : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (stateStep HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV) s = some final ∧
      final.mem = (stateScatter s HOut K V BK BV
        (roundedState cfg s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV)).mem := by
  exact FPStages.state_software_fp32 rest cfg HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV s

specification output_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest .ieeeValue)
      (output "state" "q" "out" 0 2 1 1 1 2 1 2 2 1) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  exact FPStages.output_nonempty_rounding rest

#axiomsClean output_fp32_stage
#axiomsClean vnew_fp32_stage
#axiomsClean state_fp32_stage
#axiomsClean vnew_software_fp32
#axiomsClean output_software_fp32
#axiomsClean state_software_fp32
#axiomsClean output_nonempty_rounding
specification dkHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV scale) := by
  exact FPStages.dkHeadwise_fp32_stage M DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dkHeadwise_fp32_stage

specification dkScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV scale) := by
  exact FPStages.dkScalar_fp32_stage M DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dkScalar_fp32_stage

specification dkCorrection_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (correctionValues M s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) := by
  exact FPStages.dkCorrection_fp32_stage M HPrev dv dk t s_qk_h s_vo_h B H K V BK BV s

#axiomsClean dkCorrection_fp32_stage

specification dvHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV scale) := by
  exact FPStages.dvHeadwise_fp32_stage M DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dvHeadwise_fp32_stage

specification dvScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV scale) := by
  exact FPStages.dvScalar_fp32_stage M DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dvScalar_fp32_stage

specification dbetaHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale) s = some final ∧
      Post s final dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (dbetaValues M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale) := by
  exact FPStages.dbetaHeadwise_fp32_stage M DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale s

#axiomsClean dbetaHeadwise_fp32_stage

specification dbetaScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (scalarDbetaValue M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale))).mem := by
  exact FPStages.dbetaScalar_fp32_stage M DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dbetaScalar_fp32_stage

specification dqHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV scale) := by
  exact FPStages.dqHeadwise_fp32_stage M HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dqHeadwise_fp32_stage

specification dqScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV scale) := by
  exact FPStages.dqScalar_fp32_stage M HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dqScalar_fp32_stage

specification dkHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDkValues cfg s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dkHeadwise_software_fp32 rest cfg DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dkHeadwise_software_fp32

specification dkScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDkValues cfg s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dkScalar_software_fp32 rest cfg DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dkScalar_software_fp32

specification dkCorrection_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitCorrectionValues cfg s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) := by
  exact FPStages.dkCorrection_software_fp32 rest cfg HPrev dv dk t s_qk_h s_vo_h B H K V BK BV s

#axiomsClean dkCorrection_software_fp32

specification dvHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDvValues cfg s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dvHeadwise_software_fp32 rest cfg DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dvHeadwise_software_fp32

specification dvScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDvValues cfg s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dvScalar_software_fp32 rest cfg DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dvScalar_software_fp32

specification dbetaHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV 1) s = some final ∧
      Post s final dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDbetaValues cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dbetaHeadwise_software_fp32 rest cfg DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV s

#axiomsClean dbetaHeadwise_software_fp32

specification dbetaScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (bitScalarDbetaValue cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV (FP.Value.ofBits .fp32 0x3f800000)))).mem := by
  exact FPStages.dbetaScalar_software_fp32 rest cfg DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dbetaScalar_software_fp32

specification dqHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDqValues cfg s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dqHeadwise_software_fp32 rest cfg HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dqHeadwise_software_fp32

specification dqScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDqValues cfg s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  exact FPStages.dqScalar_software_fp32 rest cfg HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV s

#axiomsClean dqScalar_software_fp32

end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPCorrect
