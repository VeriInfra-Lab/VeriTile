import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardWhole
import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardSoftware
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (softwareSum)
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed (sequentialSum)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.Repairs.LayerNormLiger.Forward (software literal32)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def bitLoad [Inhabited (FP.Value .fp32)] (p : Parameters) (a : Acc (FP.Value .fp32) p)
    (r : RegionName) (base : Nat) (i : TileIndex [p.block]) :=
  if i.1.val < p.columns then (a.state.mem r (base+i.1.val)).read .fp32 else FP.Value.zero .fp32

def bitXhat [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (a : Acc (FP.Value .fp32) p) (row : Nat) (i : TileIndex [p.block]) :=
  if i.1.val < p.columns then FP.mul cfg
    (FP.sub cfg (bitLoad p a p.x (row*p.xStride) i) (mean p a row)) (rstd p a row)
  else FP.Value.zero .fp32

def bitWdy [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (a : Acc (FP.Value .fp32) p) (row : Nat) (i : TileIndex [p.block]) :=
  FP.mul cfg (bitLoad p a p.weight 0 i) (bitLoad p a p.dy (row*p.dyStride) i)

def bitC1 [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (a : Acc (FP.Value .fp32) p) (row : Nat) :=
  FP.div cfg (sequentialSum cfg ⟨0,by simp⟩ Bool.false (fun i =>
    FP.mul cfg (bitXhat cfg p a row i) (bitWdy cfg p a row i)) PUnit.unit)
    (literal32 literal p.columns)

def bitC2 [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (a : Acc (FP.Value .fp32) p) (row : Nat) :=
  FP.div cfg (sequentialSum cfg ⟨0,by simp⟩ Bool.false (bitWdy cfg p a row) PUnit.unit)
    (literal32 literal p.columns)

def bitDX [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (a : Acc (FP.Value .fp32) p) (row : Nat) (i : TileIndex [p.block]) :=
  FP.mul cfg (FP.sub cfg (bitWdy cfg p a row i)
    (FP.add cfg (FP.mul cfg (bitXhat cfg p a row i) (bitC1 cfg literal p a row))
      (bitC2 cfg literal p a row))) (rstd p a row)

def bitUpdate [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (p : Parameters) (row : Nat) (a : Acc (FP.Value .fp32) p) : Acc (FP.Value .fp32) p where
  state := (TileShape.allIndices [p.block]).foldl (fun s i =>
    if i.1.val < p.columns then s.write p.dx (row*p.dxStride+i.1.val)
      (.mk .fp32 (bitDX cfg literal p a row i)) else s) a.state
  dw := fun i => FP.add cfg (a.dw i)
    (FP.mul cfg (bitLoad p a p.dy (row*p.dyStride) i) (bitXhat cfg p a row i))
  db := fun i => FP.add cfg (a.db i) (bitLoad p a p.dy (row*p.dyStride) i)

theorem update_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) :
    update (software rest cfg literal rsqrt32) (softwareSum cfg) p = bitUpdate cfg literal p := by
  funext row a
  have hw : wdy (software rest cfg literal rsqrt32) p a row = bitWdy cfg p a row := by
    funext i
    simp [wdy, load, padding, bitWdy, bitLoad, software, literal32,
      VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
      VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
      FP.Software.scalarFP32, FP.Software.fp32Context]
  simp only [update, dxValue, c1, c2, hw]
  simp [xhat, load, padding, cast, binary, denom, stored,
    bitUpdate, bitDX, bitC1, bitC2, bitXhat, bitWdy, bitLoad,
    software, softwareSum, literal32,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

/-- Complete source-loop recurrence over explicit rounded scalar operations;
no unused algebra callback or opaque sum can affect these accumulators. -/
theorem accumulated_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    accumulated (software rest cfg literal rsqrt32) (softwareSum cfg) p s =
      foldRange (bitUpdate cfg literal p) (start p s) (stop p s) 1
        ⟨s,fun _ => FP.Value.zero .fp32,fun _ => FP.Value.zero .fp32⟩ := by
  unfold accumulated
  rw [update_software]
  congr 1
  simp [initial, zero, software, literal32]

#axiomsClean update_software
#axiomsClean accumulated_software
end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward
