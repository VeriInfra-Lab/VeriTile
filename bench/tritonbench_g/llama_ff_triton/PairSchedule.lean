import bench.tritonbench_g.llama_ff_triton.IO
import bench.tritonbench_g.llama_ff_triton.Proofs.PairSource

/-! Pure full-grid source schedules, shared by pair IO and its connection lemmas. -/
namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.PairExecution
open VeriTile.Triton FP.Structural Execution Interface
noncomputable section
variable {α : Type} [Inhabited α]

def originalJob (a : Arithmetic α) (p : Parameters) (pid : Nat) : Invocation α :=
  job a.model (source p) pid Bool.true

def candidateJob (a : Arithmetic α) (p : Parameters) (pid : Nat) : Invocation α :=
  job a.model (PairSource.candidate p) pid Bool.true

def candidateSchedule (a : Arithmetic α) (p : Parameters) : List (Invocation α) :=
  (List.range (Grid.count p)).map (candidateJob a p)

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.PairExecution
