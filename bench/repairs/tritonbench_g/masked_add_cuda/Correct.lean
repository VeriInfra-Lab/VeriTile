import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MaskedAddCuda.Correct
open VeriTile Triton
open scoped LogicalIO
noncomputable section

/-- The regional source implements the in-place conditional update on named
logical vectors. All launch coordinates remain quantified by the IO judgment;
vetoed and out-of-range gradient cells remain in the exact exterior frame. -/
specification masked_add_kernel_correctness (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    Spec.Real (ReaderIO.view storage n alpha ⊨ fun xs _ _ j => if xs.mask j = 0 then xs.gradient j + xs.data j * alpha else xs.gradient j) :=
  ReaderProof.correct storage n alpha

#axiomsClean masked_add_kernel_correctness
end
end VeriTile.Bench.Repairs.MaskedAddCuda.Correct
