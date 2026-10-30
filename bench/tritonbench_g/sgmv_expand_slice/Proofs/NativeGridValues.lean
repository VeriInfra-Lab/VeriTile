import bench.tritonbench_g.sgmv_expand_slice.Proofs.NativeTileValues
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridTile

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section

def cellUpdate (ops : Native.Primitives) (abi : SourceABI) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (r c : Nat) (old : Native.Word (scalar abi.output)) :=
  Native.finish ops (profile abi)
    (Native.accumulate ops (profile abi) shape segment input ((r-segment.start)/32) ((c-shape.slice)/32)
      ⟨(r-segment.start)%32,Nat.mod_lt _ (by decide)⟩
      ⟨(c-shape.slice)%32,Nat.mod_lt _ (by decide)⟩) old

theorem value_law (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (fits : SegmentFits shape segment) (live : segment.active)
    (positive : 0 < shape.columns) (slice : p.slice = shape.slice)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns) :
    SourceGrid.ValueLaw (algebra rest ops (profile abi)) (dot ops (weight abi.weight)) abi p shape segment
      (inputWords abi input) (weightWords abi input) (pack (scalar abi.output))
      (cellUpdate ops abi shape segment input) := by
  intro s previous memory output i hi
  rw [tile_values rest ops zeros abi p shape segment input s memory previous output fits live positive slice capacity i hi]
  have active := TileCoordinates.active_cell p shape segment (s.pids 0) memory.columns slice live i hi
  simp only [Native.segmentStep,if_pos active,cellUpdate]

#axiomsClean value_law
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
