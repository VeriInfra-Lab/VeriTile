import bench.repairs.tritonbench_g.add_example.Contract

/-! The identical checked contiguous FP32 domain is reused without widening it.
The Python wrapper spells out these checks before empty_like or launching.
This reuse includes dynamic non-Tensor rejection and empty-pointer handling. -/
namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous
export VeriTile.Bench.Repairs.AddExample.Contiguous
  (TensorMetadata ArgumentMetadata InputContract InputError checkTensorInputs
   checkInputs AdmittedArguments)
/-- Independent mathematical formula. -/
def formula (xs ys : Nat → ℝ) (i : Nat) : ℝ := xs i + ys i

end VeriTile.Bench.Repairs.VectorAddition.Contiguous
