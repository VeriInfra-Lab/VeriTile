import bench.pretrain_kernels.tasks.local_fused_unpermute_dmerging_probs.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
open FP.Structural
open scoped VeriTile.Spec

abbrev Rules := Spec.Assumptions FP.GuardedFragment

/-- Primitive numerical models are explicit. No whole-program callback runs. -/
def original {α : Type} (decode : α → ℝ) (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) :
    ProgramIO.GuardedProgram (IO.FP.io α) (fun vals => IO.FP.admissible vals ∧ IO.FP.metadataAdmissible decode vals ∧ Allowed vals) D where
  jobs := [{ kernel := Kernels.original, backend := M, pids := fun _ => 0 }]
  outputPrecision := [.bf16]

def candidate {α : Type} (decode : α → ℝ) (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) :
    ProgramIO.GuardedProgram (IO.FP.io α) (fun vals => IO.FP.admissible vals ∧ IO.FP.metadataAdmissible decode vals ∧ Allowed vals) D where
  jobs := [{ kernel := Kernels.candidate, backend := M, pids := fun _ => 0 }]
  outputPrecision := [.bf16]

/-- Unsolved equivalence under the displayed input/rule domains.
Choose primitive models bound to the stated backends before solving this goal. -/
def equivalenceProblem {α : Type} [Inhabited α] (decode : α → ℝ) (M N : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) (R : Rules) : Prop :=
  original decode M Allowed D ≡[R] candidate decode N Allowed D

end
end VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
