import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RealWhole
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedMetadata

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
open VeriTile.Triton FP.Structural
noncomputable section

theorem prepared_result (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (strides : PreparedMemory.Strides)
    (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ) (entry : Nat → Nat → ℝ)
    (raw : List RawSegment) (extraLengths : List Int) (segments : List Segment)
    (validated : validateSegments shape raw = some segments)
    (xlayout : GridStride.Contiguous shape.tokens shape.inner strides.inputRow strides.inputInner)
    (wlayout : SnapshotLayout.WeightContiguous shape.adapters shape.columns shape.inner
      strides.weightAdapter strides.weightColumn strides.weightInner)
    (ylayout : GridStride.Contiguous shape.outputRows shape.outputColumns strides.outputRow strides.outputColumn)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns) :
    ∃ final, runSourceJobs (SourceBody.install (algebra rest) dot) abi (PreparedMemory.arguments shape strides)
        (jobs shape.columns segments) (PreparedMemory.state abi shape X W entry raw extraLengths) = some final ∧
      SourceSuffix.OutputMemory abi (PreparedMemory.arguments shape strides) shape final
        (mathematicalResult shape X W abi.addInputs segments entry) ∧
      SourceGrid.ReadFrame (PreparedMemory.arguments shape strides)
        (PreparedMemory.state abi shape X W entry raw extraLengths) final := by
  apply source_result rest abi (PreparedMemory.arguments shape strides) shape segments X W entry
    (PreparedMemory.state abi shape X W entry raw extraLengths)
  · exact PreparedMemory.inputs abi shape strides X W entry raw extraLengths xlayout wlayout
  · exact PreparedMemory.output abi shape strides X W entry raw extraLengths ylayout
  · exact PreparedMemory.validated_metadata abi shape strides X W entry raw extraLengths segments validated
  · exact PreparedMemory.private_regions shape strides
  · exact (validated_segments_fit shape raw segments validated).2
  · rfl
  · exact capacity
  · exact ylayout

#axiomsClean prepared_result
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
