import bench.tritonbench_g.llama_ff_triton.Proofs.PairExecution
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.PairIO
open VeriTile.Triton FP.Structural Execution Interface
noncomputable section
variable {α : Type} [Inhabited α]

/-- The existing candidate changes only the output row-index multiplication.
Every CTA uses the repaired masked source; no launch, array or read footprint
is replaced by an expected-value callback. -/
def variant (a : Arithmetic α) (g : GateInstructions α) (p : Parameters) :
    ArrayProgramIO.Variant (programIO Bool.false a g p) where
  schedule := fun _ => PairExecution.candidateSchedule a p
  observe := (programIO Bool.false a g p).observe

def original (a : Arithmetic α) (g : GateInstructions α) (p : Parameters) :=
  ArrayProgramIO.Implementation.original (programIO Bool.false a g p) byteDomain

def candidate (a : Arithmetic α) (g : GateInstructions α) (p : Parameters) :=
  ArrayProgramIO.Implementation.ofVariant (variant a g p) byteDomain

/-- Complete selected grid: both actual schedules are safe and succeed,
all output cells agree, and both retain the full exterior typed-cell frame.
The input domain is exactly the retained half/int8 array domain. -/
theorem equivalent (a : Arithmetic α) (g : GateInstructions α) (p : Parameters) :
    (programIO Bool.false a g p).EquivalentOn (variant a g p) byteDomain := by
  intro xs hd bounds bounded initial binding
  have bytes := (input_reads Bool.false a g p xs initial binding).bytes hd
  have safe := program_safe Bool.false a g p xs bounds bounded initial bytes
  obtain ⟨final, ran, outputs, frame⟩ := Grid.whole Bool.false a g p initial bytes
  have runOriginal : runSchedule ((programIO Bool.false a g p).schedule xs) initial = some final :=
    (program_run Bool.false a g p xs initial).trans ran
  have exterior := output_frame Bool.false a g p xs initial final frame
  refine ⟨safe, (PairExecution.complete_safe a g p xs bounds initial).mp safe,
    final, final, runOriginal, ?_, ?_, exterior, exterior⟩
  · exact (PairExecution.complete_run a g p xs initial).trans runOriginal
  · intro o j active
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.PairIO
