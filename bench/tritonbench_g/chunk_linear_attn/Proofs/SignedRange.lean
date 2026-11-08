import bench.tritonbench_g.chunk_delta_fwd.Proofs.Sources.Range
import VeriTile.Meta.StatementAudit

/-! A natural iteration clock for an actual signed range. The source index
remains signed, including negative mirrored indices; no source loop is replaced
by a natural loop. The invariant counts exactly the reached body executions. -/
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem clock (M : Algebra α) (index : RegName) (body : List ComputeStmt)
    (offset : Int) (stop : Nat) (I : Nat → State α → Prop)
    (advance : ∀ n, n < stop → ∀ s, I n s →
      ∃ t, run M body (s.setReg index .int [] (fun _ => offset + (n : Int))) = some t ∧ I (n+1) t)
    (start : Nat) (bound : start ≤ stop) (s : State α) (initial : I start s) :
    ∃ t, rangeInt M index (offset + (start : Int)) (offset + (stop : Int)) 1 body s = some t ∧
      I stop t := by
  by_cases active : start < stop
  · obtain ⟨next, ran, maintained⟩ := advance start active s initial
    obtain ⟨final, finished, invariant⟩ := clock M index body offset stop I advance
      (start+1) (by omega) next maintained
    refine ⟨final, ?_, invariant⟩
    rw [rangeInt]
    simp only [Nat.one_ne_zero, ↓reduceIte, show offset + (start : Int) < offset + (stop : Int) by omega,
      ran, Option.bind_some]
    have successor : offset + (start : Int) + (1 : Nat) = offset + ((start+1 : Nat) : Int) := by omega
    rw [successor]
    exact finished
  · have last : start = stop := by omega
    subst start
    refine ⟨s, ?_, initial⟩
    rw [rangeInt]
    simp
termination_by stop - start
decreasing_by omega

#axiomsClean clock
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange
