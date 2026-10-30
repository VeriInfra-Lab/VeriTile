import bench.pretrain_kernels.tasks.local_mla_fa4_hd256_nonabsorb_varlen_fwd.IO

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_varlen_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
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

end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_varlen_fwd
