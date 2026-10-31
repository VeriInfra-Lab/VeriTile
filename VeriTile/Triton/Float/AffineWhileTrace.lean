import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.FP.Structural.AffineWhile
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

/-- Successful small-step execution of an ordinary while guard and body. -/
inductive Trace (M : Algebra α) (guard : Op .bool []) (body : List ComputeStmt) :
    State α → State α → Prop where
  | stop {s} (guardFalse : evalOp M none guard s=some (fun _ => Bool.false)) : Trace M guard body s s
  | next {s middle final} (guardTrue : evalOp M none guard s=some (fun _ => Bool.true))
      (ran : run M body s=some middle) (tail : Trace M guard body middle final) :
      Trace M guard body s final

def guard (counter bound : RegName) : Op .bool [] :=
  .lt .nat .nil (.ref .nat [] counter) (.ref .nat [] bound)

def NormalizedCounter (s : State α) (name : RegName) (value : Nat) : Prop :=
  s.setReg name .nat [] (fun _ => value)=s

omit [Inhabited α] in
theorem normalized_set (s : State α) (name : RegName) (value : Nat) :
    NormalizedCounter (s.setReg name .nat [] (fun _ => value)) name value := by
  simp [NormalizedCounter,State.setReg]
  funext dtype shape register
  by_cases h : register=name <;> simp [h]

omit [Inhabited α] in
theorem normalized_read (s : State α) (name : RegName) (value : Nat)
    (h : NormalizedCounter s name value) :
    s.regs .nat [] name=some (fun _ => value) := by
  have read := congrArg (fun t => t.regs .nat [] name) h
  simpa only [State.setReg_same] using read.symm

/-- The compiler's affine-loop lowering preserves successful ordinary while
execution, including every final register. The local postcondition checks only
counter progress and invariant bound; no result or whole-kernel equality is
an assumption. The source initializer and final increment establish the
normalized-counter property rather than merely a value in one register bank. -/
theorem range_to_trace (M : Algebra α) (counter bound : RegName) (stride stop : Nat)
    (positive : 0<stride) (body : List ComputeStmt)
    (preserves : ∀ n (s t : State α), NormalizedCounter s counter n →
      s.regs .nat [] bound=some (fun _ => stop) → run M body s=some t →
      NormalizedCounter t counter (n+stride) ∧ t.regs .nat [] bound=some (fun _ => stop))
    (start : Nat) (s final : State α) (counterReady : NormalizedCounter s counter start)
    (boundReady : s.regs .nat [] bound=some (fun _ => stop))
    (ran : range M counter start stop stride body s=some final) :
    Trace M (guard counter bound) body s final := by
  have aux : ∀ distance current (s final : State α), stop-current=distance →
      NormalizedCounter s counter current →
      s.regs .nat [] bound=some (fun _ => stop) →
      range M counter current stop stride body s=some final →
      Trace M (guard counter bound) body s final := by
    intro distance
    induction distance using Nat.strong_induction_on with
    | h distance ih =>
      intro current state final measure normalized boundValue ran
      have counterValue := normalized_read state counter current normalized
      by_cases active : current<stop
      · have trueGuard : evalOp M none (guard counter bound) state=some (fun _ => Bool.true) := by
          simp [guard,evalOp_unfold,numeric,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,natLt,counterValue,boundValue,active]
          funext point
          change decide (current<stop)=Bool.true
          simp [active]
        rw [range,if_neg (by omega),if_pos active,show state.setReg counter .nat [] (fun _ => current)=state from normalized] at ran
        cases bodyRan : run M body state with
        | none => simp [bodyRan] at ran
        | some middle =>
          rw [bodyRan] at ran
          have nextReady := preserves current state middle normalized boundValue bodyRan
          exact Trace.next trueGuard bodyRan
            (ih (stop-(current+stride)) (by omega) (current+stride) middle final rfl
              nextReady.1 nextReady.2 ran)
      · have falseGuard : evalOp M none (guard counter bound) state=some (fun _ => Bool.false) := by
          simp [guard,evalOp_unfold,numeric,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,natLt,counterValue,boundValue,active]
          funext point
          change decide (current<stop)=Bool.false
          simp [active]
        rw [range,if_neg (by omega),if_neg active] at ran
        have equal : state=final := Option.some.inj ran
        subst final
        exact Trace.stop falseGuard
  exact aux _ start s final rfl counterReady boundReady ran

#axiomsClean normalized_set
#axiomsClean normalized_read
#axiomsClean range_to_trace
end
end VeriTile.Triton.FP.Structural.AffineWhile
