import bench.repairs.tritonbench_g.softmax_optimize.IO
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.SoftmaxIO
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- The complete previous regional judgment is unchanged by removal of its
single input/output channel wrappers, including arbitrary original program IDs. -/
theorem logical_softmax_iff (Y X : RegionName) (M N : Nat) :
    (logicalIO Y X M N ⊨ fun xs j =>
      Real.exp (xs j) / ∑ i : Fin N, Real.exp (xs i)) ↔
    (io Y X M N).RegionalImplements (target Y X M N) := Iff.rfl

theorem logical_correct (Y X : RegionName) (M N : Nat)
    (separate : Y≠X) (positive : 0<N) :
    logicalIO Y X M N ⊨ fun xs j =>
      Real.exp (xs j) / ∑ i : Fin N, Real.exp (xs i) :=
  (logical_softmax_iff Y X M N).mpr (IOCorrect.correct Y X M N separate positive)

#axiomsClean logical_softmax_iff
#axiomsClean logical_correct
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.SoftmaxIO
