import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceReal
import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceObservation

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Math
open VeriTile.Triton FP.Structural Schedule
noncomputable section
set_option maxHeartbeats 6000000

/-- Real projection of the host Q constant. The production FP theorem retains
the host binary64 power/multiply and FP32 literal operations separately. -/
def project (p : Parameters) : Parameters := {p with factor:=RealSpec.factor p.query p.channels}

theorem grid_readback (rest : Algebra ℝ) (p : Parameters) (positive : 0< p.width)
    (block : 0< p.blockRows) (channels : 0< p.channels) (blocks : 0< p.blocks)
    (entry : State ℝ) (w : Program) (member : w∈Schedule.programs 0 (p.batches*p.blocks))
    (finite : FiniteDomain (Grid.parameters (project p) w) (Grid.atProgram entry w))
    (i : Lane p) (active : Grid.active p w i) :
    (Grid.model (algebra rest) primitives (project p) positive
      (Schedule.programs 0 (p.batches*p.blocks)) entry).mem outRegion (Grid.offset p w i)=
      .mk .int (RealSpec.nearestAway
        (127*(RealSpec.factor p.query p.channels*raw (Grid.parameters (project p) w) (Grid.atProgram entry w) i)/
          blockMaximum (Grid.parameters (project p) w) (Grid.atProgram entry w))) := by
  refine (Grid.readback (algebra rest) primitives (project p) positive block channels blocks _
    (Schedule.unique _ _) entry w member i active).trans ?_
  exact (source_readback (algebra rest) primitives (Grid.parameters (project p) w) positive
    (Grid.atProgram entry w) i active).symm.trans
    (real_readback rest (Grid.parameters (project p) w) positive (Grid.atProgram entry w) rfl finite i active)

theorem grid_scale (rest : Algebra ℝ) (p : Parameters) (positive : 0< p.width)
    (entry : State ℝ) (w : Program) (member : w∈Schedule.programs 0 (p.batches*p.blocks))
    (finite : FiniteDomain (Grid.parameters (project p) w) (Grid.atProgram entry w)) :
    (Grid.model (algebra rest) primitives (project p) positive
      (Schedule.programs 0 (p.batches*p.blocks)) entry).mem scaleRegion w.index=
      .mk .real ((RealSpec.target p.query p.channels
        (logicalBlock (Grid.parameters (project p) w) (Grid.atProgram entry w))).2) := by
  rw [Grid.scale_readback (algebra rest) primitives (project p) positive _ (Schedule.unique _ _) entry w member]
  unfold savedScale
  rw [bad_zero rest (Grid.parameters (project p) w) positive (Grid.atProgram entry w) rfl finite]
  simp only [↓reduceIte]
  rw [scale_ideal rest (Grid.parameters (project p) w) (Grid.atProgram entry w) rfl finite]
  rfl

#axiomsClean grid_readback
#axiomsClean grid_scale
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Math
