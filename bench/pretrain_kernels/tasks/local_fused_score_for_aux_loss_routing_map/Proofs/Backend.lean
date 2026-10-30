import bench.pretrain_kernels.tasks.local_fused_score_for_aux_loss_routing_map.IO

namespace VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss_routing_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
open FP.Structural

/-- Unsolved connection from the explicit lowering to a supplied native source
schedule. The native implementation must use actual source invocations.
This goal carries execution, safety, typed outputs and the exterior frame. -/
def originalBackendBindingProblem {α : Type} [Inhabited α] (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop)
    (native : ProgramIO.Implementation (IO.FP.io α)
      (fun vals => IO.FP.admissible vals ∧ Allowed vals) .exactCells) : Prop :=
  ProgramIO.Implementation.Equivalent
    { schedule := fun _ => [{ kernel := Kernels.original, backend := M, pids := fun _ => 0 }]
      observe := fun _ _ => id } native

def candidateBackendBindingProblem {α : Type} [Inhabited α] (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop)
    (native : ProgramIO.Implementation (IO.FP.io α)
      (fun vals => IO.FP.admissible vals ∧ Allowed vals) .exactCells) : Prop :=
  ProgramIO.Implementation.Equivalent
    { schedule := fun _ => [{ kernel := Kernels.candidate, backend := M, pids := fun _ => 0 }]
      observe := fun _ _ => id } native

/-- Additional finite-domain connection required by the Real specialization. -/
def originalRealProjectionProblem {α : Type} [Inhabited α] (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) : Prop :=
  ProgramIO.Implementation.Equivalent
    (⟨fun _ => [{ kernel := Kernels.original, backend := M, pids := fun _ => 0 }], fun _ _ => id⟩ :
      ProgramIO.Implementation (IO.FP.io α) (fun vals => IO.FP.admissible vals ∧ Allowed vals) .exactCells)
    ⟨fun _ => [{ kernel := Kernels.realOriginal, backend := M, pids := fun _ => 0 }], fun _ _ => id⟩

end
end VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss_routing_map
