import bench.repairs.tritonbench_g.vector_addition_custom.Kernels
import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Contract
import bench.repairs.tritonbench_g.vector_addition_custom.IO
import bench.tritonbench_g.vector_addition_custom.Proofs.Real
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.ReaderCorrect
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- Public IO data, shared with the optional flat corollary. -/
abbrev addIO := VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.addIO

/-- Regional IO for the canonical source. Both read-only inputs may observe
overlapping storage. The shared judgment proves successful source execution,
actual active-access safety, the independent sum and the exact output frame. -/
theorem add_kernel_correctness (n : Nat) :
    Spec.Real (addIO n ⊨ᵣ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.RegionalImplements.intro _ ?_ ?_
  · intro bounds s hx hy hout _
    exact VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_traceSafe
      "x" "y" "out" n blockSize bounds s hx hy hout
  · intro s xs ys hx hy
    obtain ⟨final, he, hv, hf⟩ := VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_region_run
      "x" "y" "out" n blockSize s xs ys hx hy
    exact ⟨final, he, hv, fun r i outside _ => hf r i outside⟩

#stmtSurfaceSubset add_kernel_correctness ⊆
  [Spec.Real, addIO, MaskedKernelIO₂.RegionalImplements, MaskedKernelIO₂.B]
#specNonCircular formula avoiding [add_kernel, originalKernel, commutedKernel]
#axiomsClean add_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.ReaderCorrect
