import bench.tritonbench_g.matrix_reduction.Proofs.RealHost
import bench.tritonbench_g.matrix_reduction.Proofs.Real
import bench.tritonbench_g.matrix_reduction.Proofs.TypedShape

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- Complete mathematical production wrapper: both source dtype keys,
allocation and launch metadata, the original shared kernel, independent row
maxima, Torch reference/printing/assertion, failures and the None return.
Ordinary Torch contracts are explicit; no floating/native theorem is used. -/
specification load_reduce_real_function_correctness (M N : Nat) (dtype : String)
    (ops : (storage : TypedSource.Storage) → RealHost.Ops storage M N)
    (alloc : ∀ storage, RealHost.AllocationContracts storage M N (ops storage))
    (torch : ∀ storage, RealHost.TorchContracts storage M N (ops storage)) :
    VeriTile.Spec.Real (RealHost.FunctionSpec M N dtype ops
      (RealHost.loadReduce M N dtype ops)) :=
  RealHost.load_reduce_outcomes M N dtype ops alloc torch

#axiomsClean load_reduce_real_function_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixReduction
