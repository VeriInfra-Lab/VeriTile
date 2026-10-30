import bench.tritonbench_g.cosine_compute.Proofs.ReaderReal
import bench.tritonbench_g.cosine_compute.Contract
import bench.tritonbench_g.cosine_compute.IO
import bench.tritonbench_g.cosine_compute.Proofs.Mathematical_Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CosineCompute.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

abbrev cosineIO (n : Nat) : MaskedKernelIO₁ := IO.cosine n

/-- Kernel-only Real cosine: finite flat execution, active-lane bounds and values,
masked tails and the exterior frame. No host execution premise is required. -/
specification cosine_kernel_correctness (n : Nat) :
    Spec.Real (cosineIO n ⊨ fun xs i => Real.cos (xs i)) :=
  ReaderProof.cosine_kernel_correctness n

#specNonCircular mathematical avoiding [kernel,originalKernel,candidateKernel]
#axiomsClean cosine_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.CosineCompute.Correct
