import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Correct
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open scoped BigOperators VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Complete batch × 64 source launch and supplied in-place Y. The sentinel
preserves its row; other admitted rows include all K terms, runtime scaling,
source dtype conversions and independent per-address atomic orders. -/
specification kernel_real (h : Call) (indices : Nat → Int) (orders : Orders h) :
    Spec.Real (IO.logicalIO h indices orders ⊨{IO.logicalDomain h indices} fun xs batch row =>
      ofFloat h.outputType (let old := toFloat h.outputType (xs.previous batch row)
        if indices batch.val = -1 then old else old+h.scaling*∑ k : Fin h.K,
          toFloat (dtype h.storage) (xs.input batch k) *
            toFloat (dtype h.storage) (xs.selectedWeight batch row k))) :=
  IO.logical_implements h indices orders

#axiomsClean kernel_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Correct
