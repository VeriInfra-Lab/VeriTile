import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DiagSsmTriton.SignedLoop
open VeriTile.Triton FP.Structural
noncomputable section

/-- The actual signed loop binds Int counters. Natural iteration numbers here
index an invariant, without replacing the loop or its counter register. -/
theorem invariant {A : Type} [Inhabited A] (M : Algebra A) (index : RegName)
    (stop : Nat) (body : List ComputeStmt) (P : Nat → State A → Prop)
    (stepBody : ∀ n s, n<stop → P n s → ∃ t,
      run M body (s.setReg index .int [] (fun _=>(n : Int)))=some t ∧ P (n+1) t)
    (start : Nat) (bound : start≤stop) (s : State A) (initial : P start s) :
    ∃ t, rangeInt M index (start : Int) (stop : Int) 1 body s=some t ∧ P stop t := by
  rw [rangeInt]
  simp only [Nat.one_ne_zero,if_false]
  by_cases active : start<stop
  · have less : (start : Int)<stop := by exact_mod_cast active
    simp only [less,if_true]
    obtain ⟨next,ran,preserved⟩:=stepBody start s active initial
    rw [ran]
    have bound' : start+1≤stop := by omega
    simpa only [Nat.cast_add,Nat.cast_one] using
      invariant M index stop body P stepBody (start+1) bound' next preserved
  · have same : start=stop := by omega
    subst start
    simp only [lt_self_iff_false,if_false,Option.some.injEq]
    exact ⟨s,rfl,initial⟩
termination_by stop-start
decreasing_by omega

#axiomsClean invariant
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SignedLoop
