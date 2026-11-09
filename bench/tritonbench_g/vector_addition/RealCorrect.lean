import bench.tritonbench_g.vector_addition.Proofs.ReaderCorrect

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- IO.lean declares the shared source, both input windows, output window
and common active mask. -/
abbrev addIO (n : Nat) : MaskedKernelIO₂ := Interface.addIO n

/-- Canonical-source Real addition through the regional IO contract. The
shared judgment includes active access safety, successful execution, all active
outputs and the exact exterior frame. Read-only input aliases remain allowed. -/
specification add_kernel_correctness (n : Nat) :
    Spec.Real (addIO n ⊨ᵣ fun xs ys i => xs i + ys i) :=
  ReaderCorrect.add_kernel_correctness n

#stmtSurfaceSubset add_kernel_correctness ⊆
  [Spec.Real, addIO, MaskedKernelIO₂.RegionalImplements, MaskedKernelIO₂.B]
#specNonCircular formula avoiding [add_kernel, originalKernel, commutedKernel]
#axiomsClean add_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.Correct
