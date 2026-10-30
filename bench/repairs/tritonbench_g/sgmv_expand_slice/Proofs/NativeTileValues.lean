import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeAccumulation
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.TileCoordinates
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.OutputValues

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

theorem tile_values (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (s : State Raw)
    (memory : SourceMemory.Inputs abi p shape s (inputWords abi input) (weightWords abi input))
    (previous : Nat → Nat → Native.Word (scalar abi.output))
    (output : SourceSuffix.OutputMemory abi p shape s (fun r c => pack (scalar abi.output) (previous r c)))
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (slice : p.slice = shape.slice) (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (i : TileIndex [32,32])
    (hi : SourceSuffix.active p
      (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i) :
    SourceSuffix.outputValues (algebra rest ops (profile abi)) abi p
      (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) s
      (SourceExecution.accumulated (algebra rest ops (profile abi)) (dot ops (weight abi.weight))
        abi p s segment.start (segment.adapter.getD 0) segment.length) i =
      pack (scalar abi.output) (Native.segmentStep ops (profile abi) shape segment input previous
        (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)
        (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)) := by
  have hr := Nat.lt_of_lt_of_le hi.1 (fits.output live)
  have hc : SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i <
      shape.outputColumns := by have h := hi.2; rw [slice,memory.columns] at h; omega
  have active := TileCoordinates.active_cell p shape segment (s.pids 0) memory.columns slice live i hi
  rw [SourceSuffix.outputValues_active _ _ _ _ _ _ i hi]
  simp only [SourceExecution.accumulated]
  rw [accumulated_matches rest ops zeros abi p shape segment input s memory fits live positive]
  rw [output _ _ hr hc]
  have hl : (algebra rest ops (profile abi)).fp32Load = id := rfl
  simp only [hl,ite_self,id_eq]
  rw [finish_matches,Native.segmentStep,if_pos active]
  simp only [←slice]
  simp only [TileCoordinates.row_div,TileCoordinates.row_mod,
    TileCoordinates.column_div,TileCoordinates.column_mod]

#axiomsClean tile_values
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
