import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile.Triton FP.Structural Direct Direct.FPValues
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Full batch/output-row schedule and full K-term GEMV, plus initial Y when
ADD_INPUTS is selected. Input axes retain the actual EVEN_K read width and
gathered weights; signed skip metadata, source safety and frame are unchanged. -/
specification correct (h : Call) :
    Spec.Real (PublicIO.gemvLogicalIO h ⊨{PublicIO.logicalValid} fun xs batch row =>
      (∑ k : Fin h.K, xs.inputAt batch k * xs.weight batch row k) +
        if h.add then xs.previous batch row else 0) :=
  PublicIO.logical_implements h

#axiomsClean correct
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
