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
    (k : Nat) (hk : k ≤ (tripCount p))
    (h : Invariant M carrier ops plans rowScale p initial k current) :
    ∃ final, KernelExecution.rangeInt M carrier ops plans "k" (k : Int) ((tripCount p) : Int) 1 (body p) current = some final ∧
      Invariant M carrier ops plans rowScale p initial (tripCount p) final := by
  have aux : ∀ n k (current : State F), (tripCount p) - k = n → k ≤ (tripCount p) →
      Invariant M carrier ops plans rowScale p initial k current →
      ∃ final, KernelExecution.rangeInt M carrier ops plans "k" (k : Int) ((tripCount p) : Int) 1 (body p) current = some final ∧
        Invariant M carrier ops plans rowScale p initial (tripCount p) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < (tripCount p)
      · let next := bodyExecution M carrier ops plans rowScale p initial current k h
        have hi := bodyInvariant M carrier ops plans rowScale p initial current k h
        obtain ⟨final, he, hf⟩ := ih (tripCount p - (k + 1)) (by omega)
          (k + 1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        rw [KernelExecution.rangeInt, if_pos (by simpa only [Int.ofNat_lt] using And.intro (by decide : 0 < (1 : Nat)) hlt), next.property]
        change KernelExecution.rangeInt M carrier ops plans "k" ((k+1 : Nat) : Int) ((tripCount p) : Int) 1 (body p) next.val = some final
        exact he
      · have heq : k = (tripCount p) := by omega
        subst k
        refine ⟨current, ?_, h⟩
        rw [KernelExecution.rangeInt, if_neg (by omega)]
  exact aux _ _ _ rfl hk h

#axiomsClean loopExecution
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
