import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPSoftware
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardDK
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardDV
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardDBeta
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardDQ
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false
/-! Concrete bit-pattern backward contracts. Every primitive add/mul/sub
rounds through ScalarOps; both reduction axes use the explicit sequential
profile. The software scale specialization is one, while generic backend
contracts retain arbitrary scale. No output formula depends on `rest`. -/
def bitLoadKey [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32)) (R : RegionName)
    (t stride K BK : Nat) (i : Fin BK) : FP.Value .fp32 :=
  id
    (word s R (keyAddress s t stride K BK (i,PUnit.unit)) (keyIndex s BK i < K))

def bitLoadValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32)) (R : RegionName)
    (t stride V BV : Nat) (i : Fin BV) : FP.Value .fp32 :=
  id
    (word s R (valueAddress s t stride V BV (i,PUnit.unit)) (valueIndex s BV i < V))

def bitLoadBeta [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32)) (R : RegionName)
    (headwise : Bool) (t stride T V BV : Nat) (i : Fin BV) : FP.Value .fp32 :=
  if headwise then bitLoadValue cfg s R t stride V BV i
  else id ((s.mem R (s.pids 2 * T + t)).read .real)

def bitLoadBackState [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32)) (R : RegionName)
    (K V BK BV : Nat) (i : TileIndex [BK,BV]) : FP.Value .fp32 :=
  id (word s R (backStateAddress s K V BK BV i)
    (keyIndex s BK i.1 < K ∧ valueIndex s BV i.2.1 < V))

def bitDhValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32)) (DHPrev q do_ : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BK,BV] → FP.Value .fp32 := fun i =>
  FP.add cfg (bitLoadBackState cfg s DHPrev K V BK BV i)
    (FP.mul cfg
      (FP.mul cfg (bitLoadKey cfg s q t s_qk_h K BK i.1)
        (scale))
      (bitLoadValue cfg s do_ t s_vo_h V BV i.2.1))

def bitRawDvValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32)) (DHPrev q do_ k : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BV] → FP.Value .fp32 :=
  sequentialSum cfg (shape := [BK,BV]) ⟨0, by simp⟩ Bool.false (fun i =>
    FP.mul cfg (bitDhValues cfg s DHPrev q do_ t s_qk_h s_vo_h K V BK BV scale i)
      (bitLoadKey cfg s k t s_qk_h K BK i.1))

def bitDkValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (DHPrev q do_ v beta : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BK] → FP.Value .fp32 :=
  let sums := sequentialSum cfg (shape := [BK,BV]) ⟨1, by simp⟩ Bool.false (fun i =>
    FP.mul cfg (bitDhValues cfg s DHPrev q do_ t s_qk_h s_vo_h K V BK BV scale i)
      (FP.mul cfg (bitLoadValue cfg s v t s_vo_h V BV i.2.1)
        (bitLoadBeta cfg s beta headwise t s_vo_h T V BV i.2.1)))
  fun i => id (sums i)

def bitDvValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (DHPrev q do_ k beta : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BV] → FP.Value .fp32 := fun i =>
  id (FP.mul cfg
    (bitRawDvValues cfg s DHPrev q do_ k t s_qk_h s_vo_h K V BK BV scale i)
    (bitLoadBeta cfg s beta headwise t s_vo_h T V BV i.1))

def bitRawDbetaValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (DHPrev q do_ k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BV] → FP.Value .fp32 := fun i =>
  FP.mul cfg (bitRawDvValues cfg s DHPrev q do_ k t s_qk_h s_vo_h K V BK BV scale i)
    (bitLoadValue cfg s v t s_vo_h V BV i.1)

def bitDbetaValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (DHPrev q do_ k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BV] → FP.Value .fp32 := fun i =>
  id (bitRawDbetaValues cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale i)

def bitScalarDbetaValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (DHPrev q do_ k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (scale : FP.Value .fp32) : FP.Value .fp32 :=
  id
    (sequentialSum cfg (shape := [BV]) ⟨0, by simp⟩ Bool.false
      (bitRawDbetaValues cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale) PUnit.unit)

def bitDqValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (HPrev k v beta do_ : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : FP.Value .fp32) : TileIndex [BK] → FP.Value .fp32 :=
  let products : TileIndex [BK,BV] → FP.Value .fp32 := fun i =>
    FP.mul cfg
      (FP.add cfg (bitLoadBackState cfg s HPrev K V BK BV i)
        (FP.mul cfg (bitLoadKey cfg s k t s_qk_h K BK i.1)
          (FP.mul cfg (bitLoadValue cfg s v t s_vo_h V BV i.2.1)
            (bitLoadBeta cfg s beta headwise t s_vo_h T V BV i.2.1))))
      (bitLoadValue cfg s do_ t s_vo_h V BV i.2.1)
  let sums := sequentialSum cfg (shape := [BK,BV]) ⟨1, by simp⟩ Bool.false products
  fun i => id (FP.mul cfg (sums i)
    (scale))

def bitCorrectionValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) : TileIndex [BK] → FP.Value .fp32 :=
  let products : TileIndex [BK,BV] → FP.Value .fp32 := fun i =>
    FP.mul cfg
      (id (word s dv
        (outputAddress s t s_vo_h B H V BV i.2) (valueIndex s BV i.2.1 < V)))
      (bitLoadBackState cfg s HPrev K V BK BV i)
  let sums := sequentialSum cfg (shape := [BK,BV]) ⟨1, by simp⟩ Bool.false products
  fun i => id (FP.sub cfg
    (id (word s dk
      (keyOutputAddress s t s_qk_h B H K BK i) (keyIndex s BK i.1 < K))) (sums i))


specification dkHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDkValues cfg s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dkHeadwise_fp32_stage (sequentialFP32 rest cfg) DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dkHeadwise_software_fp32

specification dkScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDkValues cfg s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dkScalar_fp32_stage (sequentialFP32 rest cfg) DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dkScalar_software_fp32

specification dkCorrection_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitCorrectionValues cfg s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) := by
  obtain ⟨final, he, hp⟩ := dkCorrection_fp32_stage (sequentialFP32 rest cfg) HPrev dv dk t s_qk_h s_vo_h B H K V BK BV s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dkCorrection_software_fp32

specification dvHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDvValues cfg s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dvHeadwise_fp32_stage (sequentialFP32 rest cfg) DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dvHeadwise_software_fp32

specification dvScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDvValues cfg s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dvScalar_fp32_stage (sequentialFP32 rest cfg) DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dvScalar_software_fp32

specification dbetaHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV 1) s = some final ∧
      Post s final dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDbetaValues cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dbetaHeadwise_fp32_stage (sequentialFP32 rest cfg) DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dbetaHeadwise_software_fp32

specification dbetaScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (bitScalarDbetaValue cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV (FP.Value.ofBits .fp32 0x3f800000)))).mem := by
  obtain ⟨final, he, hm⟩ := dbetaScalar_fp32_stage (sequentialFP32 rest cfg) DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_⟩
  have hv : rawDbetaValues (sequentialFP32 rest cfg) s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV 1 =
      bitRawDbetaValues cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV
        (FP.Value.ofBits .fp32 0x3f800000) := by
    funext i
    simp [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context]
  have hs : scalarDbetaValue (sequentialFP32 rest cfg) s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV 1 =
      bitScalarDbetaValue cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV
        (FP.Value.ofBits .fp32 0x3f800000) := by
    unfold scalarDbetaValue bitScalarDbetaValue
    rw [hv]
    rfl
  simpa only [hs] using hm

#axiomsClean dbetaScalar_software_fp32

specification dqHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDqValues cfg s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dqHeadwise_fp32_stage (sequentialFP32 rest cfg) HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dqHeadwise_software_fp32

specification dqScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDqValues cfg s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000)) := by
  obtain ⟨final, he, hp⟩ := dqScalar_fp32_stage (sequentialFP32 rest cfg) HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [loadKey, loadValue, loadBeta, loadBackState, dhValues, rawDvValues, dkValues, dvValues, rawDbetaValues, dbetaValues, scalarDbetaValue, dqValues, correctionValues, bitLoadKey, bitLoadValue, bitLoadBeta, bitLoadBackState, bitDhValues, bitRawDvValues, bitDkValues, bitDvValues, bitRawDbetaValues, bitDbetaValues, bitScalarDbetaValue, bitDqValues, bitCorrectionValues, sequentialFP32, maskedInput, word, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

#axiomsClean dqScalar_software_fp32
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
