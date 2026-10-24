import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RealWhole

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.RealExecution
open VeriTile.Triton FP.Structural RealBackend
noncomputable section

/-- Complete raw source jobs, independent K-term contraction, and read frame. -/
theorem kernel_real (rest : Algebra ℝ) (abi : SourceABI) (_inputType : abi.input=.single)
    (_weightType : abi.weight=.half) (_outputType : abi.output=.single) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ)
    (entry : Nat → Nat → ℝ) (s : State ℝ)
    (memory : SourceMemory.Inputs abi p shape s X W)
    (output : SourceSuffix.OutputMemory abi p shape s entry)
    (metadata : SourceGrid.MetadataList p 0 segments s) (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment ∈ segments, SegmentFits shape segment) (slice : p.slice = shape.slice)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    ∃ final, runSourceJobs (SourceBody.install (algebra rest) dot) abi p (jobs shape.columns segments) s = some final ∧
      SourceSuffix.OutputMemory abi p shape final
        (mathematicalResult shape X W abi.addInputs segments entry) ∧
      SourceGrid.ReadFrame p s final :=
  RealBackend.source_result rest abi p shape segments X W entry s memory output metadata separation fits slice capacity layout

#axiomsClean kernel_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.RealExecution
