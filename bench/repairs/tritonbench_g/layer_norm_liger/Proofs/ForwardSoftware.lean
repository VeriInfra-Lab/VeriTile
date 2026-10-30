import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardContract
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPSoftware
namespace VeriTile.Bench.Repairs.LayerNormLiger.Forward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum softwareSum)
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed (sequentialSum)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Source literal conversion is an explicit binding; exact +0 and 1 are fixed. -/
def literal32 (other : ℝ → FP.Value .fp32) (x : ℝ) : FP.Value .fp32 :=
  if x=0 then FP.Value.zero .fp32 else if x=1 then FP.Value.ofBits .fp32 0x3f800000 else other x
/-- Concrete separately rounded ScalarOps and ascending-lane reductions.
The rsqrt instruction and conversion of other real source literals remain
explicit primitive bindings. No hardware implementation of these is assumed. -/
def software (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    Algebra (FP.Value .fp32) :=
  { VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software rest cfg with
    literal := fun _ _ x => literal32 literal x
    unary := fun pr op x => match pr,op with
      | some .fp32, .rsqrt => rsqrt32 x
      | _,_ => rest.unary pr op x
    reduceSum := fun _ {_} ax keep xs => sequentialSum cfg ax keep xs }

def bitInput [Inhabited (FP.Value .fp32)] (p : Parameters) (s : State (FP.Value .fp32))
    (r : RegionName) (base : Nat) : TileIndex [p.block] → FP.Value .fp32 := fun i =>
  if i.1.val < p.columns then (s.mem r (base+i.1.val)).read .fp32 else FP.Value.zero .fp32

def bitMean [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) : FP.Value .fp32 :=
  FP.div cfg (sequentialSum cfg ⟨0, by simp⟩ Bool.false
    (bitInput p s p.x (s.pids 0*p.xStride)) PUnit.unit) (literal32 literal (p.columns : ℝ))
def bitCentered [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) (i : TileIndex [p.block]) : FP.Value .fp32 :=
  FP.sub cfg (bitInput p s p.x (s.pids 0*p.xStride) i) (bitMean cfg literal p s)
def bitVariance [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) : FP.Value .fp32 :=
  FP.div cfg (sequentialSum cfg ⟨0, by simp⟩ Bool.false (fun i : TileIndex [p.block] =>
    if i.1.val < p.columns then
      FP.mul cfg (bitCentered cfg literal p s i) (bitCentered cfg literal p s i)
    else FP.Value.zero .fp32) PUnit.unit)
    (literal32 literal (p.columns : ℝ))
def bitRstd [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (p : Parameters) (s : State (FP.Value .fp32)) :=
  rsqrt32 (FP.add cfg (bitVariance cfg literal p s) (literal32 literal p.epsilon))
def bitY [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (p : Parameters) (s : State (FP.Value .fp32))
    (i : TileIndex [p.block]) :=
  FP.add cfg (FP.mul cfg (FP.mul cfg (bitCentered cfg literal p s i)
    (bitRstd cfg literal rsqrt32 p s)) (bitInput p s p.weight 0 i)) (bitInput p s p.bias 0 i)

theorem input_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) (r : RegionName) (base : Nat) :
    input (software rest cfg literal rsqrt32) p s r base = bitInput p s r base := by
  funext i
  simp [input, bitInput, padding, software, literal32,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem software_values [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    stored (software rest cfg literal rsqrt32)
      (meanValue (software rest cfg literal rsqrt32) (softwareSum cfg) p s) =
        .mk .fp32 (bitMean cfg literal p s) ∧
    stored (software rest cfg literal rsqrt32)
      (rstdValue (software rest cfg literal rsqrt32) (softwareSum cfg) p s) =
        .mk .fp32 (bitRstd cfg literal rsqrt32 p s) ∧
    (∀ i, stored (software rest cfg literal rsqrt32)
      (yValue (software rest cfg literal rsqrt32) (softwareSum cfg) p s i) =
        .mk .fp32 (bitY cfg literal rsqrt32 p s i)) := by
  simp only [stored, meanValue, rstdValue, varianceValue, centered, yValue, denominator,
    xValues, wValues, bValues, input_software]
  simp [stored, meanValue, rstdValue, varianceValue, centered, yValue, denominator,
    xValues, wValues, bValues, input_software, input, padding, bitMean, bitRstd, bitVariance,
    bitCentered, bitY, bitInput, software, softwareSum, literal32,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

#axiomsClean software_values
end
end VeriTile.Bench.Repairs.LayerNormLiger.Forward
