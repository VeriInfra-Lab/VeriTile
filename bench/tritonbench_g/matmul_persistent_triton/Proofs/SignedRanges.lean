import VeriTile.Triton.Float.StructuralSafetyLoops
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SignedRanges
open VeriTile.Triton FP.Structural
noncomputable section

/-- Natural mathematical counters embedded in the actual signed source loop.
No clipping of negative runtime indices is used by this lemma. -/
theorem safe_nat {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (name : RegName) (stop stride : Nat)
    (body : List ComputeStmt) (I : Nat → State α → Prop)
    (bodySafe : ∀ q,q<stop → ∀ t,I q t →
      Safety.traceSafeList M bounds body (t.setReg name .int [] (fun _ => (q : Int))))
    (preserved : ∀ q,q<stop → ∀ t,I q t → ∀ u,
      run M body (t.setReg name .int [] (fun _ => (q : Int)))=some u → I (q+stride) u)
    (start : Nat) (s : State α) (initial : I start s) :
    Safety.rangeIntSafe M bounds name (start : Int) (stop : Int) stride body s := by
  rw [Safety.rangeIntSafe_unfold]
  by_cases hz : stride=0
  · simp [hz]
  simp only [hz,if_false]
  by_cases hc : start<stop
  · have hi : (start : Int)<(stop : Int) := by exact_mod_cast hc
    rw [if_pos hi]
    refine ⟨bodySafe start hc s initial,?_⟩
    cases ran : run M body (s.setReg name .int [] (fun _ => (start : Int))) with
    | none => trivial
    | some t =>
      simpa only [Nat.cast_add] using safe_nat M bounds name stop stride body I bodySafe preserved
        (start+stride) t (preserved start hc s initial t ran)
  · simp only [show ¬(start : Int)<(stop : Int) by exact_mod_cast hc,if_false]
termination_by stop-start
decreasing_by omega

#axiomsClean safe_nat
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SignedRanges
