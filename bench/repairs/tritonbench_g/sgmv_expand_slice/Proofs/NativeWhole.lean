import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeGridValues
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridProgram

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section

/-- Full execution of the unchanged candidate quotation across every batch
and PID, connected to the independently stated native recurrence. The entry
predicates describe private snapshots and are constructed by the host layer. -/
theorem source_result (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (input : Native.Inputs (profile abi)) (entry : Nat → Nat → Native.Word (scalar abi.output))
    (s : State Raw) (memory : SourceMemory.Inputs abi p shape s (inputWords abi input) (weightWords abi input))
    (output : SourceSuffix.OutputMemory abi p shape s (fun r c => pack (scalar abi.output) (entry r c)))
    (metadata : SourceGrid.MetadataList p 0 segments s) (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment ∈ segments, SegmentFits shape segment) (slice : p.slice = shape.slice)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    ∃ final, runSourceJobs (SourceBody.install (algebra rest ops (profile abi)) (dot ops (weight abi.weight)))
        abi p (jobs shape.columns segments) s = some final ∧
      SourceSuffix.OutputMemory abi p shape final
        (fun r c => pack (scalar abi.output) (Native.result ops (profile abi) shape segments input entry r c)) ∧
      SourceGrid.ReadFrame p s final := by
  obtain ⟨final,he,ho,hf⟩ := SourceGrid.run_segments (algebra rest ops (profile abi)) (dot ops (weight abi.weight))
    abi p shape segments (inputWords abi input) (weightWords abi input) (pack (scalar abi.output))
    (fun segment => cellUpdate ops abi shape segment input)
    (fun batch segment hs live positive => value_law rest ops zeros abi {p with batch := batch} shape segment input
      (fits segment hs) live positive slice capacity)
    separation fits slice capacity layout 0 s entry memory output metadata
  refine ⟨final,he,?_,hf⟩
  simpa only [SourceGrid.formulaStep,cellUpdate,Native.result,Native.segmentStep] using ho

#axiomsClean source_result
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
