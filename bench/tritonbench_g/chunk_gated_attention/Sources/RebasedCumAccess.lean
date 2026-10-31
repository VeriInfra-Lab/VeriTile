import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumProgram
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealCumExecution

/-! Pure cumulative-tile geometry and scatter readback. Execution is proved
directly from the repaired source in Proofs/CumulativeRealExecution. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumAccess
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram
open ChunkCumsumVector.RealStep (pointer inputTile chunkPrefix address active)
open ChunkCumsumVector (lowerTriTile)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def tileBase (p : Layout) (l : Launch) (c : Nat) :=
  l.head*p.headStride+c*p.BT*p.timeStride+l.feature*p.BS*p.channelStride

def tileLayout (p : Layout) (l : Launch) (c : Nat) : Layout :=
  {p with T := p.BT, S := p.BS, headStride := tileBase p l c}

def localLaunch : Launch := ⟨0,1⟩

theorem all_active (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) :
    RealCumExecution.tileActive (tileLayout p l c) localLaunch 0 i := by
  exact ⟨by simpa [RealCumExecution.tileActive,active,tileLayout,localLaunch] using i.1.isLt,
    by simpa [RealCumExecution.tileActive,active,tileLayout,localLaunch] using i.2.1.isLt⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumAccess
