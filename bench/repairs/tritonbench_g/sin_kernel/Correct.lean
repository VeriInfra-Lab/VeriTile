import bench.repairs.tritonbench_g.sin_kernel.Proofs.RealReader

namespace VeriTile.Bench.Repairs.SinKernel.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

def sinIO (n : Nat) : MaskedKernelIO₁ := IO.sinIO n

/-- Kernel-only Real sine: finite flat execution, active-lane bounds and values,
masked tails and the exterior frame. No host execution premise is required. -/
specification sin_kernel_correctness (n : Nat) :
    Spec.Real (sinIO n ⊨ fun xs i => Real.sin (xs i)) :=
  ReaderProof.correct n

#specNonCircular formula avoiding [kernel,originalKernel,candidateKernel]
#axiomsClean sin_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.Correct
