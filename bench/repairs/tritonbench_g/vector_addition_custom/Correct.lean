import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.ReaderCorrect

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- Public IO data, shared with the optional flat corollary. -/
abbrev addIO := VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.addIO

/-- Regional IO for the canonical source. Both read-only inputs may observe
overlapping storage. The shared judgment proves successful source execution,
actual active-access safety, the independent sum and the exact output frame. -/
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
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Correct
