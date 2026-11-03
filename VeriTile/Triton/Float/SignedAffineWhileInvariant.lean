import VeriTile.Triton.Float.AffineWhileTrace
import VeriTile.Meta.StatementAudit
namespace VeriTile.Triton.FP.Structural.AffineWhile
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

def signedGuard (counter bound : RegName) : Op .bool [] :=
  .lt .int .nil (.ref .int [] counter) (.ref .int [] bound)
def NormalizedSignedCounter (s : State α) (name : RegName) (value : Int) : Prop :=
  s.setReg name .int [] (fun _ => value)=s
omit [Inhabited α] in
theorem normalized_signed_set (s : State α) (name : RegName) (value : Int) :
    NormalizedSignedCounter (s.setReg name .int [] (fun _ => value)) name value := by
  simp [NormalizedSignedCounter,State.setReg]
  funext dtype shape register
  by_cases h : register=name <;> simp [h]
omit [Inhabited α] in
theorem normalized_signed_read (s : State α) (name : RegName) (value : Int)
    (h : NormalizedSignedCounter s name value) :
    s.regs .int [] name=some (fun _ => value) := by
  have read := congrArg (fun t => t.regs .int [] name) h
  simpa only [State.setReg_same] using read.symm

/-- The same control-flow connection with a source-proved local loop invariant.
The invariant may describe data registers and memory, but progress and the
while guard are checked explicitly at every successful body execution. -/
theorem signed_range_to_trace_invariant (M : Algebra α) (counter bound : RegName) (stride : Nat) (stop : Int)
    (positive : 0<stride) (body : List ComputeStmt) (I : Int → State α → Prop)
    (control : ∀ n s, I n s → NormalizedSignedCounter s counter n ∧
      s.regs .int [] bound=some (fun _ => stop))
    (preserves : ∀ n s t, I n s → run M body s=some t → I (n+stride) t)
    (start : Int) (s final : State α) (initial : I start s)
    (ran : rangeInt M counter start stop stride body s=some final) :
    Trace M (signedGuard counter bound) body s final := by
  have aux : ∀ distance current (s final : State α), (stop-current).toNat=distance →
      I current s → rangeInt M counter current stop stride body s=some final →
      Trace M (signedGuard counter bound) body s final := by
    intro distance
    induction distance using Nat.strong_induction_on with
    | h distance ih =>
      intro current state final measure inv ran
      have ready := control current state inv
      have normalized := ready.1
      have boundValue := ready.2
      have counterValue := normalized_signed_read state counter current normalized
      by_cases active : current<stop
      · have trueGuard : evalOp M none (signedGuard counter bound) state=some (fun _ => Bool.true) := by
          simp [signedGuard,evalOp_unfold,numericLt,counterValue,boundValue]
          funext point
          change decide (current<stop)=Bool.true
          simp [active]
        rw [rangeInt,if_neg (by omega),if_pos active,show state.setReg counter .int [] (fun _ => current)=state from normalized] at ran
        cases bodyRan : run M body state with
        | none => simp [bodyRan] at ran
        | some middle =>
          rw [bodyRan] at ran
          exact Trace.next trueGuard bodyRan
            (ih ((stop-(current+stride)).toNat) (by omega) (current+stride) middle final rfl
              (preserves current state middle inv bodyRan) ran)
      · have falseGuard : evalOp M none (signedGuard counter bound) state=some (fun _ => Bool.false) := by
          simp [signedGuard,evalOp_unfold,numericLt,counterValue,boundValue]
          funext point
          change decide (current<stop)=Bool.false
          simp [active]
        rw [rangeInt,if_neg (by omega),if_neg active] at ran
        have equal : state=final := Option.some.inj ran
        subst final
        exact Trace.stop falseGuard
  exact aux _ start s final rfl initial ran

#axiomsClean signed_range_to_trace_invariant
end
end VeriTile.Triton.FP.Structural.AffineWhile
