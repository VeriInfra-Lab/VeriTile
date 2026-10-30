import bench.tritonbench_g.lora_expand_gemv.Proofs.FPShape
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPComplete

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPCorrect
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete arbitrary-length accepted output loop, in all half/bfloat input
pairs and three floating output formats. Every BLOCK_K product, padded lane,
and node of the supplied valid reduction plan is retained. No reassociation,
finite-input restriction or input/output separation premise is introduced.
The original active-body and ADD_INPUTS=false/CAST_TYPE=false cut remains;
host dispatch, other output formats, native schedule and lowering are not claimed. -/
specification lora_expand_full_machine_correctness
    (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial : State Word)
    (hBN : 0 < p.blockRows) (hKB : p.inner ≤ p.blockInner)
    (hsep : p.output ≠ p.weight) (hcn : 0 < p.outputStride) :
    ∃ final, FP.Structural.exec (backend rest cfg plans) (source profile out p) initial = some final ∧
      Post cfg plans profile out p initial final ∧
      (∀ k : Fin p.inner, some (⟨k.val, by omega⟩ : Fin p.blockInner) ∈
        (plans p.blockInner).tree.leaves) := by
  obtain ⟨final,he,hp⟩ := correctness rest cfg plans profile out p initial hBN hcn hsep
  refine ⟨final,he,hp,?_⟩
  intro k
  apply (plans p.blockInner).valid.mem_iff.mpr
  simp

#axiomsClean lora_expand_full_machine_correctness
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPCorrect
