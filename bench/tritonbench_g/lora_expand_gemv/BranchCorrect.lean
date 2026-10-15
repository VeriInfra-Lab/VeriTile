import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchComplete
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchShape

/-!
Complete per-program LoRA expand source execution, including signed sentinel,
all EVEN_K/ADD_INPUTS/CAST_TYPE choices, and the full dynamic output loop.

Six input/weight profiles enumerate the wrapper's declared input and weight
formats; output storage is explicitly limited to fp16, bf16 and fp32. Python
does not enumerate its output or index dtype, so this contract alone does not
close the unrestricted caller dtype domain. Integer width/ABI, memory safety,
host grid composition, and native reduction/configuration refinement remain
outside this software execution theorem.
-/
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.BranchCorrect
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
open Branches
noncomputable section
local instance branchCorrectWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Every declared kernel branch and every output iteration executes to the
independent scalar/tree/memory recurrence, with an exact whole-memory frame.
No output equation, intermediate equality, backend law or alias restriction
is assumed. -/
specification lora_expand_all_branches_machine_correctness
    (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial : State Word)
    (hBN : 0 < p.blockRows) :
    ∃ final, FP.Structural.exec (engine rest cfg plans) (source profile out p) initial = some final ∧
      final.mem = expectedMemory cfg plans profile out p initial ∧
      (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o) :=
  correctness rest cfg plans profile out p initial hBN

#axiomsClean lora_expand_all_branches_machine_correctness
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.BranchCorrect
