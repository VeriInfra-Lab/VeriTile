import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.PublicIO
import bench.tritonbench_g.bgmv_shrink_kernel.IO

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IO
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open scoped BigOperators VeriTile.Triton.LogicalDomainIO
noncomputable section

theorem logical_iff (h : Call) (indices : Nat → Int) (orders : Orders h) :
    (logicalIO h indices orders ⊨{logicalDomain h indices} fun xs b row =>
      ofFloat h.outputType (let old := toFloat h.outputType (xs.previous b row)
        if indices b.val = -1 then old else old+h.scaling*∑ k : Fin h.K,
          toFloat (dtype h.storage) (xs.input b k) *
            toFloat (dtype h.storage) (xs.selectedWeight b row k))) ↔
    (programIO Real.algebra Real.primitives h indices h.scaling orders id).ImplementsOnCells
      (domain h indices) (Mathematical.real h indices) := by
  rfl

theorem logical_implements (h : Call) (indices : Nat → Int) (orders : Orders h) :
    logicalIO h indices orders ⊨{logicalDomain h indices} fun xs b row =>
      ofFloat h.outputType (let old := toFloat h.outputType (xs.previous b row)
        if indices b.val = -1 then old else old+h.scaling*∑ k : Fin h.K,
          toFloat (dtype h.storage) (xs.input b k) *
            toFloat (dtype h.storage) (xs.selectedWeight b row k)) :=
  (logical_iff h indices orders).mpr (PublicIO.real h indices orders)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IO
