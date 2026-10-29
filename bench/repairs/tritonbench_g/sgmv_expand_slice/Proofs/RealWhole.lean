import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RealGridValues
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridProgram

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
open VeriTile.Triton FP.Structural
noncomputable section

/-- Exact Real interpretation of the same typed source, with no FP numerical
assumptions. This is not yet a claim about the separate BlockState projection
runner; that interpreter connection is stated separately in the public entry. -/
theorem source_result (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
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
      SourceGrid.ReadFrame p s final := by
  obtain ⟨final,he,ho,hf⟩ := SourceGrid.run_segments (algebra rest) dot abi p shape segments X W id
    (fun segment => cellUpdate abi shape segment X W)
    (fun batch segment hs live positive => value_law rest abi {p with batch := batch} shape segment X W
      (fits segment hs) live positive slice capacity)
    separation fits slice capacity layout 0 s entry memory output metadata
  refine ⟨final,he,?_,hf⟩
  simpa only [SourceGrid.formulaStep,cellUpdate,mathematicalResult,mathematicalStep,id_eq] using ho

#axiomsClean source_result
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
