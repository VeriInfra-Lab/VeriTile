import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardErasure

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton

/-- A complete fp16 instance with all three flags universally quantified.
It reuses the checked finite-case erasure bridge. -/
theorem fp16_symbolic_erasure_probe (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) (s : BlockState) :
    exec (Typed.homogeneousKernel .fp16 Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s =
    exec (source GradIn GradOut Out bm bn gm gn om on K D
      log causal fp16).toAlgKernel s := by
  exact typed_erased_execution .fp16 GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16 s

#axiomsClean fp16_symbolic_erasure_probe
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
