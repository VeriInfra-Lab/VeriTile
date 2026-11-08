import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Successful source while execution with a scalar integer counter. The
body is interpreted directly, and each recursive step tests the actual counter.
The signed range refinement requires canonical counter bindings and proves the
same final state; it assumes no body result, output equality or floating law. -/
namespace VeriTile.Triton.FP.Structural.ScalarInt64While
noncomputable section
variable {A : Type} [Inhabited A]

/-- Rebinding an already canonical counter to the same value is inert. -/
theorem counter_idempotent (s : State A) (counter : RegName) (value : Int) :
    (s.setReg counter .int [] (fun _=>value)).setReg counter .int [] (fun _=>value)=
      s.setReg counter .int [] (fun _=>value) := by
  cases s
  simp only [State.setReg]
  congr 1
  funext d sh name
  by_cases hn : name=counter
  · subst name
    by_cases hd : TileDType.int=d
    · subst d
      by_cases hs : []=sh
      · subst sh; simp
      · simp [hs]
    · simp [hd]
  · simp [hn]

inductive Executes (M : Algebra A) (counter : RegName) (finish : Int)
    (body : List ComputeStmt) : State A → State A → Prop
  | stop {s : State A} {current : Int}
      (read : s.regs .int [] counter=some (fun _=>current))
      (done : finish≤current) : Executes M counter finish body s s
  | next {s u t : State A} {current : Int}
      (read : s.regs .int [] counter=some (fun _=>current))
      (active : current<finish) (step : run M body s=some u)
      (rest : Executes M counter finish body u t) : Executes M counter finish body s t

/-- A canonical source counter and its actual unit increment refine the range
implementation. Clients discharge progress from their complete source body. -/
theorem of_range (M : Algebra A) (counter : RegName) (finish : Int)
    (body : List ComputeStmt)
    (progress : ∀ (current : Int) (s u : State A),
      s.setReg counter .int [] (fun _=>current)=s → run M body s=some u →
      u.setReg counter .int [] (fun _=>current+1)=u)
    (current : Int) (s t : State A)
    (canonical : s.setReg counter .int [] (fun _=>current)=s)
    (ran : rangeInt M counter current finish 1 body s=some t) :
    Executes M counter finish body s t := by
  have read : s.regs .int [] counter=some (fun _=>current) := by
    conv_lhs => rw [← canonical]
    exact State.setReg_same _ _ _ _ _
  by_cases active : current<finish
  · rw [rangeInt] at ran
    simp only [show (1 : Nat)≠0 by decide,if_false,active,if_true,canonical] at ran
    cases step : run M body s with
    | none => simp [step] at ran
    | some u =>
      simp only [step,Option.bind_some] at ran
      exact .next read active step
        (of_range M counter finish body progress (current+1) u t
          (progress current s u canonical step) ran)
  · rw [rangeInt] at ran
    simp only [show (1 : Nat)≠0 by decide,if_false,active,if_false] at ran
    cases Option.some.inj ran
    exact .stop read (Int.le_of_not_gt active)
termination_by (finish-current).toNat
decreasing_by omega

#axiomsClean counter_idempotent
#axiomsClean of_range
end
end VeriTile.Triton.FP.Structural.ScalarInt64While
