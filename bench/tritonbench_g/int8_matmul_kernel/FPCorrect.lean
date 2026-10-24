import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordWitness
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.FPCorrect
open VeriTile.Triton FP.Structural WordModel
noncomputable section
/-- Full original four-field packed GEMM with arbitrary inner-loop extent.
Actual signed-byte narrowing and 32-bit wrap are part of the value model;
there is no no-overflow premise and no mathematical-Int dot fallback. -/
specification int8_matmul_word_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (p : Parameters) (s : State α)
    (hBlock : 0 < p.blockInner) (hK : p.inner = 4*(p.blockInner*p.steps))
    (hInputs : Inputs p s)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (backend rest) (source p) s = some final ∧
      Post p s final ∧
      ∀ f k, FP.IntegerDot.valid (α := α) (aValue p s f k) ∧
        FP.IntegerDot.valid (α := α) (bValue p s f k) := by
  obtain ⟨final, he, hp⟩ := correctness rest p s hw
  exact ⟨final, he, hp, dot_inputs_valid p s⟩
#axiomsClean int8_matmul_word_correctness
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.FPCorrect
