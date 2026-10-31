import bench.tritonbench_g.triton_argmax.Proofs.TypedStep

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def vectorLoop (initial : State Word) (X : RegionName) (M N K BM BN row inner : Nat)
    (positive : 0<BN) (cur : Nat) (current : Fin BM → Word × Nat) : Fin BM → Word × Nat :=
  if h : cur<N then vectorLoop initial X M N K BM BN row inner positive (cur+BN)
    (next initial X M N K BM BN row inner cur positive current) else current
termination_by N-cur
decreasing_by omega

theorem next_memory (initial s : State Word) (same : s.mem=initial.mem) (X : RegionName)
    (M N K BM BN row inner cur : Nat) (positive : 0<BN) (current : Fin BM → Word × Nat) :
    next s X M N K BM BN row inner cur positive current=
      next initial X M N K BM BN row inner cur positive current := by
  funext i
  have loads : loaded s X M N K BM BN row inner cur i=loaded initial X M N K BM BN row inner cur i := by
    funext j
    simp only [loaded,same]
  simp only [next,loads,same]

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
