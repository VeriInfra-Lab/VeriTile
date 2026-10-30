import bench.tritonbench_g.matrix_transpose.Kernels
import bench.tritonbench_g.matrix_transpose.Proofs.Indices
import bench.tritonbench_g.matrix_transpose.Proofs.Mathematical_FPBits

/-! Exact kernel-only FP16 copy. Every floating primitive is arbitrary and
unreachable. These proofs reuse the complete existing typed execution witness;
no host runner, allocation or result callback supplies a transpose value. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.KernelFP
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixTranspose

def parameters (m d : Nat) : FPBits.Parameters :=
  ⟨"matrix", "out", max d 1, 1, max m 1, 1, m, d⟩

theorem original_source (m d : Nat) :
    source "matrix" "out" m d = FPBits.source Bool.false (parameters m d) := rfl

theorem commuted_source (m d : Nat) :
    indexCommutedSource "matrix" "out" m d = FPBits.source Bool.true (parameters m d) := rfl

theorem output_injective (m d : Nat) :
    Function.Injective (FPBits.outputOffset (parameters m d)) := by
  intro a b eq
  have ha : a.2.1.val < m := a.2.1.isLt
  have hb : b.2.1.val < m := b.2.1.isLt
  have hm : 1 ≤ m := by omega
  simp only [FPBits.outputOffset, parameters, Nat.max_eq_left hm, Nat.mul_one] at eq
  have lower := congrArg (fun o => o % m) eq
  dsimp only at lower
  rw [Nat.mul_add_mod_of_lt ha, Nat.mul_add_mod_of_lt hb] at lower
  have upper : a.1.val = b.1.val :=
    Nat.mul_right_cancel (by omega : 0 < m) (by omega)
  apply Prod.ext (Fin.ext upper)
  exact Prod.ext (Fin.ext lower) (Subsingleton.elim _ _)

/-- Every typed input payload, output coordinate, and exterior cell. The
input premise is a concrete FP16 storage binding, not a numerical restriction. -/
theorem coordinates {α : Type} [Inhabited α] (M : Algebra α)
    (m d : Nat) (h : InputContract m d) (s : State α) (matrix : Nat → α)
    (input : InputCells m d s.mem matrix) :
    ∃ t, exec M (source "matrix" "out" m d) s = some t ∧
      (∀ row : Fin m, ∀ col : Fin d,
        t.mem "out" (col.val*m+row.val) = .mk .fp16 (formula matrix d row.val col.val)) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ m*d ≤ o → t.mem r o = s.mem r o) := by
  refine ⟨FPBits.execution (parameters m d) s, FPBits.execute M Bool.false (parameters m d) s, ?_, ?_⟩
  · intro row col
    have value := FPBits.readback (parameters m d) s (output_injective m d) (col,row,())
    have bound := Indices.flat_bound m d row.val col.val row.isLt col.isLt
    have read : FPBits.outputValue (parameters m d) s (col,row,()) =
        formula matrix d row.val col.val := by
      unfold FPBits.outputValue FPBits.inputOffset parameters formula
      simp only [Nat.max_eq_left h.positive.2, Nat.mul_one]
      rw [input _ bound, Cell.read_same]
    rw [read] at value
    simpa only [FPBits.outputOffset, parameters, Nat.max_eq_left h.positive.1,
      Nat.max_eq_left h.positive.2, Nat.mul_one] using value
  · intro r o outside
    apply FPBits.frame (parameters m d) s r o
    rcases outside with hn | ho
    · exact Or.inl hn
    · right
      intro i eq
      have bound : FPBits.outputOffset (parameters m d) i < m*d := by
        simpa [FPBits.outputOffset, parameters, Nat.max_eq_left h.positive.1, Nat.mul_comm] using
          Indices.flat_bound d m i.1.val i.2.1.val i.1.isLt i.2.1.isLt
      omega

/-- Both existing source variants succeed to the same complete state.
Only exact natural-number index multiplication is commuted. -/
theorem variants_execute {α : Type} [Inhabited α] (M : Algebra α)
    (m d : Nat) (s : State α) :
    ∃ t, exec M (source "matrix" "out" m d) s = some t ∧
      exec M (indexCommutedSource "matrix" "out" m d) s = some t :=
  ⟨FPBits.execution (parameters m d) s,
    FPBits.execute M Bool.false (parameters m d) s, FPBits.execute M Bool.true (parameters m d) s⟩

/-- This source's result cannot depend on any floating arithmetic backend. -/
theorem backend_independent {α : Type} [Inhabited α] (M N : Algebra α)
    (m d : Nat) (s : State α) :
    exec M (source "matrix" "out" m d) s = exec N (source "matrix" "out" m d) s :=
  FPBits.backend_independent M N Bool.false (parameters m d) s

#axiomsClean original_source
#axiomsClean commuted_source
#axiomsClean coordinates
#axiomsClean variants_execute
#axiomsClean backend_independent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatrixTranspose.KernelFP
