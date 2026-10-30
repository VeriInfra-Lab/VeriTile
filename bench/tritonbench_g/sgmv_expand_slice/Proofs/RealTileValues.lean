import bench.tritonbench_g.sgmv_expand_slice.Proofs.RealAccumulation
import bench.tritonbench_g.sgmv_expand_slice.Proofs.TileCoordinates
import bench.tritonbench_g.sgmv_expand_slice.Proofs.OutputValues

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

theorem finish_formula (rest : Algebra ℝ) (abi : SourceABI) (acc old : ℝ) :
    SourceSuffix.finish (algebra rest) abi acc old = if abi.addInputs then acc+old else acc := by
  simp [SourceSuffix.finish,SourceSuffix.castIfNeeded,algebra]

theorem tile_values (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (segment : Segment) (s : State ℝ) (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ)
    (memory : SourceMemory.Inputs abi p shape s X W) (previous : Nat → Nat → ℝ)
    (output : SourceSuffix.OutputMemory abi p shape s previous)
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (slice : p.slice = shape.slice) (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (i : TileIndex [32,32])
    (hi : SourceSuffix.active p
      (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i) :
    SourceSuffix.outputValues (algebra rest) abi p
      (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) s
      (SourceExecution.accumulated (algebra rest) dot abi p s segment.start (segment.adapter.getD 0) segment.length) i =
      mathematicalStep shape X W abi.addInputs segment previous
        (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)
        (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i) := by
  have hr := Nat.lt_of_lt_of_le hi.1 (fits.output live)
  have hc : SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i <
      shape.outputColumns := by have h := hi.2; rw [slice,memory.columns] at h; omega
  have active := TileCoordinates.active_cell p shape segment (s.pids 0) memory.columns slice live i hi
  rw [SourceSuffix.outputValues_active _ _ _ _ _ _ i hi]
  simp only [SourceExecution.accumulated]
  rw [accumulated_contraction rest abi p shape s X W memory _
    (SourceMemory.coordinates_rows p shape segment fits live (s.pids 0))
    (SourceMemory.coordinates_adapter shape segment fits live)
    (SourceMemory.coordinates_columns p shape segment memory.columns positive (s.pids 0))]
  rw [output _ _ hr hc]
  have hl : (algebra rest).fp32Load = id := rfl
  simp only [hl,ite_self,id_eq]
  rw [finish_formula,mathematicalStep,if_pos active]
  have hs : (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length (s.pids 0)).start = segment.start := rfl
  rw [hs]
  rw [TileCoordinates.gathered_row p segment (s.pids 0) i hi,
    TileCoordinates.gathered_column p segment (s.pids 0) i hi,slice]
  rfl

#axiomsClean finish_formula
#axiomsClean tile_values
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
