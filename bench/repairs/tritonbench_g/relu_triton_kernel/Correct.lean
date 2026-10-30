import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.ReluTritonKernel.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Every program owns its full active input/output tile. No pid-zero gate. -/
def reluIO (N : Nat) : MaskedKernelIO₁ := RealIO.io N

/-- Kernel-only mathematical correctness: successful execution, every active
output, safe masked accesses, and the exterior memory frame. The original
benchmark's defective gated store is not asserted correct by this theorem. -/
specification relu_kernel_correctness (N : Nat) :
    Spec.Real (reluIO N ⊨ fun xs i => max 0 (xs i)) :=
  RealIO.implements N

#stmtSurfaceSubset relu_kernel_correctness ⊆
  [Spec.Real, reluIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#axiomsClean relu_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.Correct
