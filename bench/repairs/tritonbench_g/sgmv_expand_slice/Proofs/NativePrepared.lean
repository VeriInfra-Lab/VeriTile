import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeWhole
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedMetadata

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section

def prepared (abi : SourceABI) (shape : Shape) (input : Native.Inputs (profile abi))
    (entry : Nat → Nat → Native.Word (scalar abi.output)) (raw : List RawSegment) (extraLengths : List Int) : State Raw :=
  PreparedMemory.state abi shape (inputWords abi input) (weightWords abi input)
    (fun r c => pack (scalar abi.output) (entry r c)) raw extraLengths

theorem prepared_result (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (shape : Shape) (strides : PreparedMemory.Strides)
    (input : Native.Inputs (profile abi)) (entry : Nat → Nat → Native.Word (scalar abi.output))
    (raw : List RawSegment) (extraLengths : List Int) (segments : List Segment)
    (validated : validateSegments shape raw = some segments)
    (xlayout : GridStride.Contiguous shape.tokens shape.inner strides.inputRow strides.inputInner)
    (wlayout : SnapshotLayout.WeightContiguous shape.adapters shape.columns shape.inner
      strides.weightAdapter strides.weightColumn strides.weightInner)
    (ylayout : GridStride.Contiguous shape.outputRows shape.outputColumns strides.outputRow strides.outputColumn)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns) :
    ∃ final, runSourceJobs (SourceBody.install (algebra rest ops (profile abi)) (dot ops (weight abi.weight)))
        abi (PreparedMemory.arguments shape strides) (jobs shape.columns segments)
        (prepared abi shape input entry raw extraLengths) = some final ∧
      SourceSuffix.OutputMemory abi (PreparedMemory.arguments shape strides) shape final
        (fun r c => pack (scalar abi.output) (Native.result ops (profile abi) shape segments input entry r c)) ∧
      SourceGrid.ReadFrame (PreparedMemory.arguments shape strides) (prepared abi shape input entry raw extraLengths) final := by
  apply source_result rest ops zeros abi (PreparedMemory.arguments shape strides) shape segments input entry
    (prepared abi shape input entry raw extraLengths)
  · exact PreparedMemory.inputs abi shape strides _ _ _ raw extraLengths xlayout wlayout
  · exact PreparedMemory.output abi shape strides _ _ _ raw extraLengths ylayout
  · exact PreparedMemory.validated_metadata abi shape strides _ _ _ raw extraLengths segments validated
  · exact PreparedMemory.private_regions shape strides
  · exact (validated_segments_fit shape raw segments validated).2
  · rfl
  · exact capacity
  · exact ylayout

#axiomsClean prepared_result
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
