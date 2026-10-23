import bench.repairs.tritonbench_g.add_value.Proofs.LogicalReader
import bench.repairs.tritonbench_g.add_value.IO
import bench.tritonbench_g.add_value.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The shared literal-add10 source with block size 1024.
This IO describes one program's active input/output lanes in flat memory. -/
def addValueIO (n : Nat) : MaskedKernelIO₁ := IO.addValueIO n

/-- Kernel correctness for arbitrary program IDs, including active-lane
results, masked tails, access bounds and the exterior memory frame. -/
specification puzzle1_kernel_correctness (n : Nat) :
    Spec.Real (addValueIO n ⊨ fun xs i => xs i + 10) :=
  ReaderProof.correct n

#stmtSurfaceSubset puzzle1_kernel_correctness ⊆
  [Spec.Real, addValueIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#specNonCircular mathematical avoiding [kernel, originalKernel, commutedKernel]
#axiomsClean puzzle1_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.Correct
