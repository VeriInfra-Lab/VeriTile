import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ZeroSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ZeroComplexBackward
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 600000

/-- Real zero-time IO. The static access bounds are checked against a finite
allocation even when all time windows are empty. The conclusion includes
source trace safety, successful flattened execution, exact zero output cells
and the full tagged-cell frame. -/
def Implements (kernel : ComputeKernel)
    (bounds : RegionBounds → BlockState → Prop)
    (writes : FlatAlloc → BlockState → RegionName → Nat → Prop) : Prop :=
  ∀ (A : FlatAlloc), A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ u : BlockState, bounds A.extent u →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u) = some v ∧
      (∀ r o, writes A u r o → v.mem r o = MemCell.real 0) ∧
      (∀ r o, ¬ writes A u r o → v.mem r o = (A.flattenState u).mem r o)

theorem zero_readback {v : BlockState} {r : RegionName} {o : Nat}
    (h : v.mem r o = MemCell.real 0) : v.readMem r o = 0 := by
  unfold BlockState.readMem
  rw [h]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroIO
