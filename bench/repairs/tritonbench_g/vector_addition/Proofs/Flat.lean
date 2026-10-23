import bench.repairs.tritonbench_g.vector_addition.IO
import bench.tritonbench_g.vector_addition.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.Flat
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- Optional flat-memory corollary. `MaskedKernelIO₂.Implements` requires
pairwise disjoint allocations for x, y and out; it is not the alias-permitting
Python contract. The primary regional IO theorem has no such requirement. -/
theorem add_kernel_flat_correctness (n : Nat) :
    Spec.Real (Interface.addIO n ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_flattenOk
      "x" "y" "out" n blockSize
  · intro bounds s hx hy hout _
    exact VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_traceSafe
      "x" "y" "out" n blockSize bounds s hx hy hout
  · intro s xs ys hx hy
    obtain ⟨final, he, hv, hf⟩ := VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_region_run
      "x" "y" "out" n blockSize s xs ys hx hy
    exact ⟨final, he, hv, fun r i outside _ => hf r i outside⟩

#axiomsClean add_kernel_flat_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.Flat
