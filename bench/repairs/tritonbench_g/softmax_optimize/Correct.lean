import bench.repairs.tritonbench_g.softmax_optimize.IO
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Correct
open VeriTile Triton
open SoftmaxIO
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- The complete selected row implements the independent mathematical softmax.
The public N-element windows include every full and tail tile of both passes. -/
specification softmax_row_correct (Y X : RegionName) (M N : Nat) (separate : Y ≠ X)
    (positive : 0 < N) :
    Spec.Real ((logicalIO Y X M N) ⊨ fun xs j =>
      Real.exp (xs j) / ∑ i : Fin N, Real.exp (xs i)) :=
  logical_correct Y X M N separate positive

#specNonCircular rowSoftmax avoiding [exec,kernel]
#axiomsClean softmax_row_correct
#stmtSurfaceSubset softmax_row_correct ⊆
  [Spec.Real, LogicalIO.Implements, logicalIO, RegionName, Real.exp, Finset.sum,UKernelIO.Ctx,io,UKernelIO.nOut,UKernelIO.oarity,ChanTy.carrier,UKernelIO.oty,UKernelIO.RegionalImplements]
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Correct
