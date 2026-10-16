import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPContract
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed (sequentialSum)
open VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages (literal32)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Each arithmetic step uses ScalarOps, with explicit sequential reduction.
The sqrt primitive and conversion of nontrivial literals remain named inputs. -/
def software (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    Algebra (FP.Value .fp32) :=
  { VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software rest cfg literal sqrt32 with
    unary := fun pr op x => match pr,op with
      | some .fp32, .sqrt => sqrt32 x
      | _,_ => rest.unary pr op x }

def bitX [Inhabited (FP.Value .fp32)] (p : Parameters) (s : State (FP.Value .fp32))
    (n : Nat) : FP.Value .fp32 :=
  if n<p.columns then (s.mem p.x (xBase p s+n)).read .fp32 else FP.Value.zero .fp32
def bitW [Inhabited (FP.Value .fp32)] (p : Parameters) (s : State (FP.Value .fp32))
    (n : Nat) : FP.Value .fp32 :=
  if n<p.columns then (s.mem p.weight (wBase p s+n)).read .fp32
    else s.undef .fp32 p.weight (wBase p s+n)
def bitMeanStep [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (off : Nat) (a : TileIndex [p.block] → FP.Value .fp32) :=
  fun i : TileIndex [p.block] => FP.add cfg (a i) (bitX p s (off+i.1.val))
def bitMean [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :=
  FP.div cfg (sequentialSum cfg ⟨0, by simp⟩ Bool.false
    (foldRange (bitMeanStep cfg p s) 0 p.columns p.block (fun _ => FP.Value.zero .fp32)) PUnit.unit)
    (literal32 literal (p.columns : ℝ))
def bitCentered [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) (n : Nat) :=
  if n<p.columns then FP.sub cfg (bitX p s n) (bitMean cfg literal p s) else FP.Value.zero .fp32
def bitVarStep [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) (off : Nat) (a : TileIndex [p.block] → FP.Value .fp32) :=
  fun i : TileIndex [p.block] => FP.add cfg (a i)
    (FP.mul cfg (bitCentered cfg literal p s (off+i.1.val)) (bitCentered cfg literal p s (off+i.1.val)))
def bitVariance [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :=
  FP.div cfg (sequentialSum cfg ⟨0, by simp⟩ Bool.false
    (foldRange (bitVarStep cfg literal p s) 0 p.columns p.block (fun _ => FP.Value.zero .fp32)) PUnit.unit)
    (literal32 literal (p.columns : ℝ))
def bitRstd [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (sqrt32 : FP.Value .fp32 → FP.Value .fp32) (p : Parameters) (s : State (FP.Value .fp32)) :=
  FP.div cfg (FP.Value.ofBits .fp32 0x3f800000)
    (sqrt32 (FP.add cfg (bitVariance cfg literal p s) (literal32 literal p.epsilon)))
def bitY [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (sqrt32 : FP.Value .fp32 → FP.Value .fp32) (p : Parameters) (s : State (FP.Value .fp32)) (n : Nat) :=
  FP.mul cfg (FP.mul cfg (FP.sub cfg (bitX p s n) (bitMean cfg literal p s))
    (bitRstd cfg literal sqrt32 p s)) (bitW p s n)

theorem initial_software (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32) (p : Parameters) :
    initial (software rest cfg literal sqrt32) p = (fun _ => FP.Value.zero .fp32) := by
  funext i
  simp [initial, software, literal32, VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software]

theorem mean_step_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    meanStep (software rest cfg literal sqrt32) p s=bitMeanStep cfg p s := by
  funext off a i
  simp [meanStep, xValue, bitMeanStep, bitX, software, literal32,
    VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  all_goals norm_num

theorem mean_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    meanValue (software rest cfg literal sqrt32) p s=bitMean cfg literal p s := by
  unfold meanValue
  rw [mean_step_software, initial_software]
  simp [bitMean, initial, software, literal32,
    VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem var_step_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    varStep (software rest cfg literal sqrt32) p s=bitVarStep cfg literal p s := by
  funext off a i
  simp only [varStep, centered, mean_software]
  simp [xValue, bitVarStep, bitCentered, bitX, software, literal32,
    VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  all_goals norm_num

theorem variance_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    varianceValue (software rest cfg literal sqrt32) p s=bitVariance cfg literal p s := by
  unfold varianceValue
  rw [var_step_software, initial_software]
  simp [bitVariance, initial, software, literal32,
    VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem output_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) (n : Nat) :
    yValue (software rest cfg literal sqrt32) p s n=bitY cfg literal sqrt32 p s n := by
  simp only [yValue, rstdValue, mean_software, variance_software]
  simp [xValue, wValue, bitY, bitX, bitW, bitRstd, software, literal32,
    VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages.software,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  all_goals norm_num

#axiomsClean mean_step_software
#axiomsClean mean_software
#axiomsClean var_step_software
#axiomsClean variance_software
#axiomsClean output_software
end
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
