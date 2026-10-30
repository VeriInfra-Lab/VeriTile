import bench.tritonbench_g.sgmv_expand_slice.Proofs.RealTileValues
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridTile

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
open VeriTile.Triton FP.Structural
noncomputable section

def cellUpdate (abi : SourceABI) (shape : Shape) (segment : Segment)
    (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ) (r c : Nat) (old : ℝ) :=
  let computed := contraction shape X W (segment.adapter.getD 0) r (c-shape.slice)
  if abi.addInputs then computed+old else computed

theorem value_law (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (segment : Segment) (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ)
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (slice : p.slice = shape.slice) (capacity : shape.slice+shape.columns ≤ shape.outputColumns) :
    SourceGrid.ValueLaw (algebra rest) dot abi p shape segment X W id (cellUpdate abi shape segment X W) := by
  intro s previous memory output i hi
  rw [tile_values rest abi p shape segment s X W memory previous output fits live positive slice capacity i hi]
  have active := TileCoordinates.active_cell p shape segment (s.pids 0) memory.columns slice live i hi
  simp only [mathematicalStep,if_pos active,cellUpdate,id_eq]

#axiomsClean value_law
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
