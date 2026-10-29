import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelExecution
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelNativeSpec

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelNative
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
noncomputable section

theorem complete (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (u : State Node) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    ∃ v,KernelExecution.Facts (model isa plan kind c) kind c s seed p u v ∧
      Outputs isa plan kind c s seed p u v := by
  obtain ⟨v,facts⟩ := KernelExecution.complete (model isa plan kind c) kind c s seed p u nonempty hh hk hv
  refine ⟨v,facts,?_⟩
  cases kind with
  | forward =>
    intro t e j
    rw [facts.outputs t e j,expected_lowering]
    rfl
  | backward =>
    refine ⟨?_,?_,?_⟩
    · intro t e j
      rw [facts.outputs.1 t e j,expected_lowering]
      rfl
    · intro t
      rw [facts.outputs.2.1 t,expected_lowering]
      rfl
    · intro e j
      rw [facts.outputs.2.2 e j,expected_lowering]
      rfl

#axiomsClean complete
#specNonCircular expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelNative
