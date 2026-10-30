import bench.repairs.tritonbench_g.sin_computation.Proofs.RealReader

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Kernel-only Real sine: finite flat execution, active-lane bounds and values,
masked tails and the exterior frame. No host execution premise is required. -/
specification sin_kernel_correctness (X O : RegionName) (n : Nat) :
    Spec.Real (sinIO X O n ⊨ fun xs i => Real.sin (xs i)) :=
  ReaderProof.correct X O n

#stmtSurfaceSubset sin_kernel_correctness ⊆
  [Spec.Real, RegionName, sinIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B, Real.sin]
#specNonCircular formula avoiding [kernel,originalKernel,candidateKernel]
#axiomsClean sin_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct
