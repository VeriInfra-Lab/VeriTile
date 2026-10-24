import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Geometry
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.PrepareCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.BackwardCorrect

/-! Reuse the original complete per-program Real execution and frame proofs.
Their size/separation premises follow from independent packed allocations and
the complete/short launch geometry; no original proof is copied or weakened.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.RealTiles
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DecayCumsum
noncomputable section

def forwardIO (h : Call) (p : Phase) (scale : ℝ) :=
  ForwardCorrect.originalIO .fp32 "g" "go"
    h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
def prepareIO (h : Call) (p : Phase) (scale : ℝ) :=
  PrepareCorrect.originalIO .fp32 .fp32 "q" "k" "g" "qg" "kg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale
def backwardIO (h : Call) (p : Phase) (scale : ℝ) :=
  BackwardCorrect.originalIO .fp32 .fp32 "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale

theorem forward_source (h : Call) (p : Phase) (scale : ℝ) :
    (forwardIO h p scale).kernel=(Kernels.source .forward h p scale).eraseDType := rfl
theorem prepare_source (h : Call) (p : Phase) (scale : ℝ) :
    (prepareIO h p scale).kernel=(Kernels.source .prepare h p scale).eraseDType := rfl
theorem backward_source (h : Call) (p : Phase) (scale : ℝ) :
    (backwardIO h p scale).kernel=(Kernels.source .backward h p scale).eraseDType := rfl

theorem forward_complete (h : Call) (p : Phase) (scale : ℝ) (positive : 0<h.DK) :
    StreamEmitMasked3DKernelIO₁.Implements (forwardIO h p scale)
      (fun _ _ _ xs t j => ForwardCorrect.scan p.span h.BK xs t j) := by
  exact ForwardCorrect.original_correct .fp32 "g" "go"
    h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
    (by decide) (Geometry.tile_le_width h positive)

theorem prepare_complete (h : Call) (p : Phase) (hp : p∈phases h)
    (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (prepareIO h p scale)
      (fun _ _ _ xs o t j => PrepareCorrect.values p.span h.BK scale xs o t j) := by
  exact PrepareCorrect.original_correct .fp32 .fp32 "q" "k" "g" "qg" "kg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (Geometry.tile_le_width h positive) (Geometry.phase_positive h p hp).2.1

theorem backward_complete (h : Call) (p : Phase) (hp : p∈phases h)
    (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (backwardIO h p scale)
      (fun _ _ _ xs o t j => BackwardCorrect.values p.span h.BK xs o t j) := by
  exact BackwardCorrect.original_correct .fp32 .fp32 "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (Geometry.tile_le_width h positive) (Geometry.phase_positive h p hp).2.1

#axiomsClean forward_complete
#axiomsClean prepare_complete
#axiomsClean backward_complete
end
end VeriTile.Bench.Repairs.DecayCumsum.RealTiles
