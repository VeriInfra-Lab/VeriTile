import bench.tritonbench_g.layer_norm_welfold.Proofs.FPSourceBlock
import bench.tritonbench_g.layer_norm_welfold.Proofs.FPRsqrtPrimitive
namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialFP32 sequentialSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def sourceWord [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32))
    (R : RegionName) (offset : Nat) (active : Prop) [Decidable active]
    {RBLOCK : Nat} (i : TileIndex [1,RBLOCK]) : FP.Value .fp32 :=
  if active then (s.mem R offset).read .real else s.undef .real R offset

def roundedSourceMean [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (s : State (FP.Value .fp32)) (X : RegionName) (rnumel RBLOCK : Nat) : FP.Value .fp32 :=
  let xs : TileIndex [1,RBLOCK] → FP.Value .fp32 := fun i =>
    FP.add cfg (FP.Value.zero .fp32)
      (sourceWord s X (sourceColumn i + rnumel * s.pids 0) (sourceColumn i < rnumel) i)
  let total := sequentialSum cfg (shape := [1,RBLOCK]) ⟨1, by simp⟩ Bool.false xs (⟨0, by decide⟩, PUnit.unit)
  FP.div cfg total (FP.round .fp32 cfg (rnumel : Rat))

def roundedSourceRstd [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (s : State (FP.Value .fp32))
    (X : RegionName) (rnumel RBLOCK : Nat) (mean : FP.Value .fp32) : FP.Value .fp32 :=
  let squared : TileIndex [1,RBLOCK] → FP.Value .fp32 := fun i =>
    let x := sourceWord s X (sourceColumn i + rnumel * s.pids 0) (sourceColumn i < rnumel) i
    let delta := FP.sub cfg x mean
    FP.add cfg (FP.Value.zero .fp32) (FP.mul cfg delta delta)
  let total := sequentialSum cfg (shape := [1,RBLOCK]) ⟨1, by simp⟩ Bool.false squared (⟨0, by decide⟩, PUnit.unit)
  rsqrt32 (FP.add cfg (FP.div cfg total (FP.round .fp32 cfg (rnumel : Rat))) epsilon32)

def roundedSourceY [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (X W Bias : RegionName) (rnumel RBLOCK : Nat) (mean rstd : FP.Value .fp32) : TileIndex [1,RBLOCK] → FP.Value .fp32 := fun i =>
  let active := sourceColumn i < rnumel
  let x := sourceWord s X (sourceColumn i + rnumel * s.pids 0) active i
  let weight := sourceWord s W (sourceColumn i) active i
  let bias := sourceWord s Bias (sourceColumn i) active i
  FP.add cfg (FP.mul cfg (FP.mul cfg (FP.sub cfg x mean) rstd) weight) bias

def roundedSourceMemory [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (s : State (FP.Value .fp32))
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) : State (FP.Value .fp32) :=
  let avg := roundedSourceMean cfg s X rnumel RBLOCK
  let afterMean := s.write mean (s.pids 0) (.mk .real avg)
  let inverse := roundedSourceRstd cfg rsqrt32 afterMean X rnumel RBLOCK avg
  let afterRstd := afterMean.write rstd (s.pids 0) (.mk .real inverse)
  (TileShape.allIndices [1,RBLOCK]).foldl (fun t i =>
    if sourceColumn i < rnumel then t.write Y (sourceColumn i + rnumel * s.pids 0)
      (.mk .real (roundedSourceY cfg afterRstd X W Bias rnumel RBLOCK avg inverse i)) else t) afterRstd

theorem sourceMean_profile [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (s : State (FP.Value .fp32)) (X : RegionName) (rnumel RBLOCK : Nat) :
    sourceMean (fp32WithRsqrt rest cfg rsqrt32) s X rnumel RBLOCK = roundedSourceMean cfg s X rnumel RBLOCK := by
  simp [sourceMean, roundedSourceMean, sourceLoad, sourceWord, fp32WithRsqrt,
    sequentialFP32, FP.Software.scalarFP32, FP.Software.fp32Context,
    show (0 : ℝ) ≠ 1e-05 by norm_num]

theorem sourceRstd_profile [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (s : State (FP.Value .fp32)) (X : RegionName) (rnumel RBLOCK : Nat) (mean : FP.Value .fp32) :
    sourceRstd (fp32WithRsqrt rest cfg rsqrt32) s X rnumel RBLOCK mean =
      roundedSourceRstd cfg rsqrt32 s X rnumel RBLOCK mean := by
  simp [sourceRstd, roundedSourceRstd, sourceLoad, sourceWord, fp32WithRsqrt,
    sequentialFP32, FP.Software.scalarFP32, FP.Software.fp32Context,
    show (0 : ℝ) ≠ 1e-05 by norm_num]

theorem sourceY_profile [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (s : State (FP.Value .fp32)) (X W Bias : RegionName) (rnumel RBLOCK : Nat) (mean rstd : FP.Value .fp32) :
    sourceY (fp32WithRsqrt rest cfg rsqrt32) s X W Bias rnumel RBLOCK mean rstd =
      roundedSourceY cfg s X W Bias rnumel RBLOCK mean rstd := by
  funext i
  simp [sourceY, roundedSourceY, sourceLoad, sourceWord, fp32WithRsqrt,
    sequentialFP32, FP.Software.scalarFP32, FP.Software.fp32Context]

specification source_one_block_fp32_with_rsqrt [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (mean rstd X W Bias Y : RegionName) (xnumel rnumel RBLOCK : Nat) (s : State (FP.Value .fp32))
    (hpos : 0 < rnumel) (hle : rnumel ≤ RBLOCK) :
    ∃ final, FP.Structural.exec (fp32WithRsqrt rest cfg rsqrt32)
      (sourceOneBlock mean rstd X W Bias Y xnumel rnumel RBLOCK) s = some final ∧
      final.mem = (roundedSourceMemory cfg rsqrt32 s mean rstd X W Bias Y rnumel RBLOCK).mem := by
  obtain ⟨final, he, hm⟩ := source_one_block_fp32 (fp32WithRsqrt rest cfg rsqrt32)
    mean rstd X W Bias Y xnumel rnumel RBLOCK s hpos hle
  exact ⟨final, he, by simpa only [sourceMemory, roundedSourceMemory,
    sourceMean_profile, sourceRstd_profile, sourceY_profile] using hm⟩

#axiomsClean source_one_block_fp32_with_rsqrt
end
end VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
