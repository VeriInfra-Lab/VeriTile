import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelGemmExecution

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 28000
variable {I F H : Type} [Inhabited F]

/-- Every source K iteration executes successfully with the actual incoming
accumulator. This includes zero iterations, both split counts, and the final
masked block. It does not assert any algebraic simplification of native MMA. -/
theorem loopExecution (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (rowScale : Precision) (p : Parameters) (initial current : State F)
    (k : Nat) (hk : k ≤ p.steps)
    (h : Invariant M carrier ops plans rowScale p initial k current) :
    ∃ final, KernelExecution.range M carrier ops plans "k" k p.steps 1 (body p) current = some final ∧
      Invariant M carrier ops plans rowScale p initial p.steps final := by
  have aux : ∀ n k (current : State F), p.steps - k = n → k ≤ p.steps →
      Invariant M carrier ops plans rowScale p initial k current →
      ∃ final, KernelExecution.range M carrier ops plans "k" k p.steps 1 (body p) current = some final ∧
        Invariant M carrier ops plans rowScale p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := bodyExecution M carrier ops plans rowScale p initial current k h
        have hi := bodyInvariant M carrier ops plans rowScale p initial current k h
        obtain ⟨final, he, hf⟩ := ih (p.steps - (k + 1)) (by omega)
          (k + 1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        rw [KernelExecution.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k = p.steps := by omega
        subst k
        refine ⟨current, ?_, h⟩
        rw [KernelExecution.range, if_neg (by decide), if_neg (by omega)]
  exact aux _ _ _ rfl hk h

#axiomsClean loopExecution
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
