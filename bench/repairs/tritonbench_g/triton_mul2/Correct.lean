import bench.repairs.tritonbench_g.triton_mul2.Proofs.ReaderCorrect

/-! The ordinary and in-place source kernels implement the real formula 2 * xs i.
The two commuted candidate results are retained in Misc/CandidateCorrect.lean.
The masked IO contracts include termination, active-address bounds and the
memory frame. The in-place contracts read the original values before writing
back to the same region. The host grid composition is outside these contracts. -/

namespace VeriTile.Bench.Repairs.TritonMul2.Correct

open VeriTile.Triton VeriTile.Examples
open VeriTile.Bench.TritonBenchG.TritonMul2
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Symbolic windows and tail mask for ordinary output. -/
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  IO.mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE

specification mul2_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i) :=
  ReaderCorrect.mul2_kernel_correctness in_ptr0 out_ptr n_elements BLOCK_SIZE

/-- Both roles share the same region for in-place output. -/
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  IO.mul2InplaceIO ptr n_elements BLOCK_SIZE

specification mul2_inplace_kernel_correctness
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2InplaceIO ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i) :=
  ReaderCorrect.mul2_inplace_kernel_correctness ptr n_elements BLOCK_SIZE

#stmtSurfaceSubset mul2_kernel_correctness ⊆
  [Spec.Real, RegionName, mul2IO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#stmtSurfaceSubset mul2_inplace_kernel_correctness ⊆
  [Spec.Real, RegionName, mul2InplaceIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]

#specNonCircular mathematical avoiding [sourceKernel, commutedSourceKernel]
#axiomsClean mul2_kernel_correctness
#axiomsClean mul2_inplace_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.TritonMul2.Correct
