import bench.repairs.tritonbench_g.sgmv_expand_slice.IO
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.IOReal

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
open scoped BigOperators VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Full old/new judgment equivalence, with all signed metadata including
inactive entries, the ordered overlap semantics and complete output readback. -/
theorem logical_real_iff (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs)
    (shape : Shape) (segments : List Segment)
    (f : LogicalInputs ℝ shape segments → Fin shape.outputRows → Fin shape.outputColumns → ℝ)
    (formula : f = fun xs row col =>
      (segments.foldl (fun previous segment r c =>
        if ActiveCell shape segment r c then
          let v := ∑ k ∈ Finset.range shape.inner,
            xs.inputAt r k * xs.weightAt (segment.adapter.getD 0) (c-shape.slice) k
          if abi.addInputs then v + previous r c else v
        else previous r c) xs.entryAt) row.val col.val) :
    (logicalIO (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi p shape segments
      ⊨{LogicalMetadataDomain} f) ↔
    (io (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi p shape segments).ImplementsOn
      (metadataDomain p shape segments) (fun xs => resultValues p shape (realFormula abi p shape segments xs)) := by
  have connection : (logicalIO (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
      abi p shape segments).lift f =
      (fun xs => resultValues p shape (realFormula abi p shape segments xs)) := by
    rw [formula]
    funext raw port j
    fin_cases port
    rfl
  change (io _ abi p shape segments).ImplementsOn (metadataDomain p shape segments) _ ↔ _
  rw [connection]

#axiomsClean logical_real_iff
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
