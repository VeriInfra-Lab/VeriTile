import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.AccessSafety
open VeriTile Triton FP.Structural
noncomputable section

/-- Follow the actual unit-stride loop states, reusing successful source-body
execution and its invariant. This adds no evaluator or safety callback to IO. -/
theorem range_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (stop : Nat) (body : List ComputeStmt)
    (I : Nat → State α → Prop)
    (advance : ∀ n current,n<stop → I n current →
      Safety.traceSafeList M bounds body (current.setReg idx .nat [] (fun _ => n)) ∧
      ∃ next,run M body (current.setReg idx .nat [] (fun _ => n))=some next ∧ I (n+1) next)
    (start : Nat) (s : State α) (hs : I start s) :
    Safety.rangeSafe M bounds idx start stop 1 body s := by
  have aux : ∀ distance n current,stop-n=distance → I n current →
      Safety.rangeSafe M bounds idx n stop 1 body current := by
    intro distance
    induction distance using Nat.strong_induction_on with
    | h distance ih =>
      intro n current hn hi
      by_cases active : n<stop
      · obtain ⟨safe,next,ran,inv⟩ := advance n current active hi
        rw [Safety.rangeSafe_unfold,if_neg (by decide),if_pos active,ran]
        exact ⟨safe,ih (stop-(n+1)) (by omega) (n+1) next rfl inv⟩
      · simp [Safety.rangeSafe_unfold,active]
  exact aux _ start s rfl hs
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.AccessSafety
