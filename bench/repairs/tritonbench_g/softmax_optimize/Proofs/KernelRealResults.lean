import bench.repairs.tritonbench_g.softmax_optimize.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.KernelRealResults
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
noncomputable section

/-- Independent mathematical row softmax. -/
def reference (input : Fin M → Fin N → ℝ) (row : Fin M) (column : Fin N) : ℝ :=
  Real.exp (input row column)/(∑ j : Fin N,Real.exp (input row j))

/-- Complete selected source execution over raw kernel memory, independent
row-softmax values and the full exterior row frame. -/
theorem softmax_row_correct (Y X : RegionName) (M N : Nat) (separate : Y ≠ X)
    (positive : 0 < N) (initial : BlockState) :
    Spec.Real (∃ final, exec (kernel Y X M N (Config.tile N)).eraseDType.toAlgKernel initial=some final ∧
      (∀ j : Fin N,final.readMem Y (initial.pid*N+j.val)=softmaxOptimizeFullSpec initial X N j) ∧
      RowFrame Y N initial final ∧ final.pids=initial.pids ∧
      final.numPids=initial.numPids ∧ final.undef=initial.undef) :=
  SourceIO.selected_tile_io Y X M N separate positive initial

#specNonCircular reference avoiding [exec,kernel]
#axiomsClean softmax_row_correct
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.KernelRealResults
