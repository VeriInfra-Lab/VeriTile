import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.NativeExecution
open VeriTile.Triton FP.Structural NativeBackend
noncomputable section

/-- Complete selected FP32/FP16 source jobs and independent native recurrence,
with raw memory/metadata reads, geometry, separation and original primitive laws. -/
theorem kernel_native (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (_inputType : abi.input=.single)
    (_weightType : abi.weight=.half) (_outputType : abi.output=.single) (p : KernelArgs) (shape : Shape) (segments : List Segment)
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
      SourceGrid.ReadFrame p s final :=
  NativeBackend.source_result rest ops zeros abi p shape segments input entry s memory output metadata separation fits slice capacity layout

#axiomsClean kernel_native
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.NativeExecution
