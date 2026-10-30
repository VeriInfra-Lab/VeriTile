import bench.tritonbench_g.vector_addition.IO
import bench.tritonbench_g.vector_addition.Proofs.Contract
import bench.tritonbench_g.vector_addition.Proofs.Projection
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_Real
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.ReaderCorrect
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- IO.lean declares the shared source, both input windows, output window
and common active mask. -/
abbrev addIO (n : Nat) : MaskedKernelIO₂ := Interface.addIO n

/-- Canonical-source Real addition through the regional IO contract. The
shared judgment includes active access safety, successful execution, all active
outputs and the exact exterior frame. Read-only input aliases remain allowed. -/
theorem add_kernel_correctness (n : Nat) :
    Spec.Real (addIO n ⊨ᵣ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.RegionalImplements.intro _ ?_ ?_
  · intro bounds s hx hy hout _
    exact VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_traceSafe
      "x" "y" "out" n blockSize bounds s hx hy hout
  · intro s xs ys hx hy
    obtain ⟨final, he, hv, hf⟩ := VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_region_run
      "x" "y" "out" n blockSize s xs ys hx hy
    exact ⟨final, he, hv, fun r i outside _ => hf r i outside⟩

#stmtSurfaceSubset add_kernel_correctness ⊆
  [Spec.Real, addIO, MaskedKernelIO₂.RegionalImplements, MaskedKernelIO₂.B]
#specNonCircular formula avoiding [add_kernel, originalKernel, commutedKernel]
#axiomsClean add_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.ReaderCorrect
