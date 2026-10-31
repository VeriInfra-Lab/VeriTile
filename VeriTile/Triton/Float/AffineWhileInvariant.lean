import VeriTile.Triton.Float.AffineWhileTrace
import VeriTile.Meta.StatementAudit
namespace VeriTile.Triton.FP.Structural.AffineWhile
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

/-- The same control-flow connection with a source-proved local loop invariant.
The invariant may describe data registers and memory, but progress and the
while guard are checked explicitly at every successful body execution. -/
theorem range_to_trace_invariant (M : Algebra α) (counter bound : RegName) (stride stop : Nat)
    (positive : 0<stride) (body : List ComputeStmt) (I : Nat → State α → Prop)
    (control : ∀ n s, I n s → NormalizedCounter s counter n ∧
      s.regs .nat [] bound=some (fun _ => stop))
    (preserves : ∀ n s t, I n s → run M body s=some t → I (n+stride) t)
    (start : Nat) (s final : State α) (initial : I start s)
    (ran : range M counter start stop stride body s=some final) :
    Trace M (guard counter bound) body s final := by
  have aux : ∀ distance current (s final : State α), stop-current=distance →
      I current s → range M counter current stop stride body s=some final →
      Trace M (guard counter bound) body s final := by
    intro distance
    induction distance using Nat.strong_induction_on with
    | h distance ih =>
      intro current state final measure inv ran
      have ready := control current state inv
      have normalized := ready.1
      have boundValue := ready.2
      have counterValue := normalized_read state counter current normalized
      by_cases active : current<stop
      · have trueGuard : evalOp M none (guard counter bound) state=some (fun _ => Bool.true) := by
          simp [guard,evalOp_unfold,natLt,counterValue,boundValue]
          funext point
          change decide (current<stop)=Bool.true
          simp [active]
        rw [range,if_neg (by omega),if_pos active,show state.setReg counter .nat [] (fun _ => current)=state from normalized] at ran
        cases bodyRan : run M body state with
        | none => simp [bodyRan] at ran
        | some middle =>
          rw [bodyRan] at ran
          exact Trace.next trueGuard bodyRan
            (ih (stop-(current+stride)) (by omega) (current+stride) middle final rfl
              (preserves current state middle inv bodyRan) ran)
      · have falseGuard : evalOp M none (guard counter bound) state=some (fun _ => Bool.false) := by
          simp [guard,evalOp_unfold,natLt,counterValue,boundValue]
          funext point
          change decide (current<stop)=Bool.false
          simp [active]
        rw [range,if_neg (by omega),if_neg active] at ran
        have equal : state=final := Option.some.inj ran
        subst final
        exact Trace.stop falseGuard
  exact aux _ start s final rfl initial ran

#axiomsClean range_to_trace_invariant
end
end VeriTile.Triton.FP.Structural.AffineWhile
