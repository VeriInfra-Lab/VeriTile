import bench.tritonbench_g.vector_addition.Contract
import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.add_example.Proofs.Contract
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteIndices

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous
export VeriTile.Bench.Repairs.AddExample.Contiguous
  (checkTensorInputs_iff checkInputs_iff check_tensor_arguments_iff same_numel bounded_inputs)

/-- Every lane, including the inactive tail, has exact signed-int32 controls. -/
theorem native_indices (x y : TensorMetadata) (h : InputContract x y)
    (pid lane : Nat) (hp : pid < gridCount x.numel) (hl : lane < 1024) :
    (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane).toInt =
      (pid * 1024 + lane : Nat) ∧
    VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 x.numel pid lane =
      decide (pid * 1024 + lane < x.numel) :=
  (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.source_index_arithmetic
    x.numel pid lane h.extent hp hl).2
#axiomsClean native_indices
end VeriTile.Bench.Repairs.VectorAddition.Contiguous
