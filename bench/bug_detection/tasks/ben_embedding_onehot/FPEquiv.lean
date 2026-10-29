import bench.bug_detection.tasks.ben_embedding_onehot.Contract

namespace VeriTile.Bench.BugDetection.ben_embedding_onehot
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open FP.Structural
open scoped VeriTile.Spec

abbrev Rules := Spec.Assumptions FP.GuardedFragment

/-- Primitive numerical models are explicit. No whole-program callback runs. -/
def original {α : Type} (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) :
    ProgramIO.GuardedProgram (IO.FP.io α) (fun vals => IO.FP.admissible vals ∧ Allowed vals) D where
  jobs := [{ kernel := Kernels.original, backend := M, pids := fun _ => 0 }]
  outputPrecision := [.bf16]

def candidate {α : Type} (M : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) :
    ProgramIO.GuardedProgram (IO.FP.io α) (fun vals => IO.FP.admissible vals ∧ Allowed vals) D where
  jobs := [{ kernel := Kernels.candidate, backend := M, pids := fun _ => 0 }]
  outputPrecision := [.bf16]

/-- Unsolved equivalence under the displayed input/rule domains.
Choose primitive models bound to the stated backends before solving this goal. -/
def equivalenceProblem {α : Type} [Inhabited α] (M N : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) (R : Rules) : Prop :=
  original M Allowed D ≡[R] candidate N Allowed D

def inequivalenceProblem {α : Type} [Inhabited α] (M N : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) (D : FP.Guarded.Domain α) (R : Rules) : Prop :=
  ¬ equivalenceProblem M N Allowed D R

/-- An exact output/interface mismatch is separate from refuting ≡[R]. -/
def counterexampleProblem {α : Type} [Inhabited α] (M N : Algebra α)
    (Allowed : (IO.FP.io α).Inputs → Prop) : Prop :=
  ∃ vals, IO.FP.admissible vals ∧ Allowed vals ∧ ∃ bounds initial left right,
    (IO.FP.io α).InBounds bounds vals ∧
    (IO.FP.io α).InputCells vals initial ∧
    ScheduleSafe bounds [{ kernel := Kernels.original, backend := M, pids := fun _ => 0 }] initial ∧
    ScheduleSafe bounds [{ kernel := Kernels.candidate, backend := N, pids := fun _ => 0 }] initial ∧
    runSchedule [{ kernel := Kernels.original, backend := M, pids := fun _ => 0 }] initial = some left ∧
    runSchedule [{ kernel := Kernels.candidate, backend := N, pids := fun _ => 0 }] initial = some right ∧
    ∃ offset < 64*2048, left.mem "out" offset ≠ right.mem "out" offset

end
end VeriTile.Bench.BugDetection.ben_embedding_onehot
