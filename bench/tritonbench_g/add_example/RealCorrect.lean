import bench.tritonbench_g.add_example.Proofs.LogicalReader
import bench.tritonbench_g.add_example.IO
import bench.tritonbench_g.add_example.Kernels
import bench.tritonbench_g.add_example.Proofs.Contract
import bench.tritonbench_g.add_example.Proofs.Projection
import bench.tritonbench_g.add_example.Proofs.Mathematical_Real
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddExample.Contiguous.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- One program's IO for the canonical source at the wrapper's block size. -/
def addIO (n : Nat) : MaskedKernelIO₂ := IO.addIO n

/-- The canonical source safely implements the independent addition formula
on all active lanes, with successful regional execution and the exact exterior
frame. Read-only inputs need no separation premise. -/
specification add_kernel_correctness (n : Nat) :
    Spec.Real (addIO n ⊨ᵣ fun xs ys i => xs i + ys i) :=
  ReaderProof.correct n

#stmtSurfaceSubset add_kernel_correctness ⊆
  [Spec.Real, addIO, MaskedKernelIO₂.RegionalImplements, MaskedKernelIO₂.B]
#specNonCircular formula avoiding [add_kernel, originalKernel, commutedKernel]
#axiomsClean add_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddExample.Contiguous.Correct
