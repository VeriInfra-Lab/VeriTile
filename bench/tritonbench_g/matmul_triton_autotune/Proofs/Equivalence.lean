import bench.tritonbench_g.matmul_triton_autotune.EquivIO
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Refinement

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.EquivalenceProof
open VeriTile.Triton FP.Structural Whole
open FloatingIO
noncomputable section
local instance : Inhabited Floating.Word := ⟨FP.Value.zero .fp32⟩

/-- Reuse the proved complete source simulation. The two operands are the
old source and existing safe-address candidate, rather than a native formula. -/
theorem equivalent (instructions : Floating.Instructions) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) :
    Equivalent (original instructions g ids order) (safe instructions g ids order) := by
  intro initial
  exact programs_refinement Floating.scalar (Floating.dot instructions g.config)
    g.request ids initial initial rfl

#axiomsClean equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.EquivalenceProof
