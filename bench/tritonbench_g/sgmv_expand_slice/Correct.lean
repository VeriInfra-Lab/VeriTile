import bench.tritonbench_g.sgmv_expand_slice.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Correct
open VeriTile.Triton FP.Structural
open scoped BigOperators VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Complete ordered source jobs. FP32 input, FP16 weights, FP32 in-place
output, signed metadata, original strides and only active writable cells. -/
def sgmvIO (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs)
    (shape : Shape) (segments : List Segment) :=
  PublicIO.logicalIO (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi p shape segments

/-- Ordinary K-term contraction, with optional input addition and the original
segment order for overlaps; all untouched output coordinates retain their value. -/
def sgmvFormula (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : PublicIO.LogicalInputs ℝ shape segments)
    (row : Fin shape.outputRows) (col : Fin shape.outputColumns) : ℝ :=
    (segments.foldl (fun previous segment r c =>
      if ActiveCell shape segment r c then
        let v := ∑ k ∈ Finset.range shape.inner,
          xs.inputAt r k *
            xs.weightAt (segment.adapter.getD 0) (c-shape.slice) k
        if abi.addInputs then v + previous r c else v
      else previous r c) xs.entryAt) row.val col.val

specification kernel_real (rest : Algebra ℝ) (abi : SourceABI)
    (inputType : abi.input=.single) (weightType : abi.weight=.half) (outputType : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    Spec.Real ((sgmvIO rest abi p shape segments) ⊨{PublicIO.LogicalMetadataDomain}
      sgmvFormula abi p shape segments) := by
  apply (PublicIO.logical_real_iff rest abi p shape segments (sgmvFormula abi p shape segments) rfl).mpr
  exact PublicIO.real_implements rest abi inputType weightType outputType p shape segments
    separation fits inner columns slice capacity layout

#axiomsClean kernel_real
#specNonCircular sgmvFormula avoiding [runSourceJobs,runRealJobs,FP.Structural.run,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Correct
