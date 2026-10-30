import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelForwardInit
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelBackward

import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelExecutionSpec

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
noncomputable section
variable {α : Type} [Inhabited α]
theorem complete (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    ∃ v,Facts M kind c s seed p u v := by
  cases kind with
  | forward =>
    obtain ⟨v,he,ho,hf⟩ := KernelForwardInit.source_correct M c s seed p u nonempty hh hk hv
    exact ⟨v,he,ho,hf⟩
  | backward =>
    obtain ⟨v,he,hi,hg,hl,hf⟩ := KernelBackward.source_correct M c s seed p u nonempty hh hk hv
    refine ⟨v,he,?_,hf⟩
    simpa [Output,expected,KernelReverseLoop.carry] using And.intro hi (And.intro hg hl)

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelExecution
