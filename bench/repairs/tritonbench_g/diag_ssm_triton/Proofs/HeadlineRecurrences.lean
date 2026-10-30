import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference

namespace VeriTile.Bench.Repairs.DiagSsmTriton.HeadlineRecurrences
open VeriTile.Bench.TritonBenchG.DiagSsmTriton

/-- Recurrence uniqueness connects any displayed pure state function to the
canonical independent mathematical specification. -/
theorem state_eq {α : Type} [Semiring α] (initial transition : α) (input f : Nat → α)
    (zero : f 0=initial) (step : ∀ t,f (t+1)=f t*transition+input t) (t : Nat) :
    f t=Reference.forwardState initial transition input t := by
  induction t with
  | zero => exact zero
  | succ t ih => simp only [step, Reference.forwardState, ih]

theorem carry_eq {α : Type} [Semiring α] (length : Nat) (transition : α) (gy f : Nat → α)
    (zero : f 0=0) (step : ∀ k,f (k+1)=(gy (length-1-k)+f k)*transition) (k : Nat) :
    f k=Reference.reverseCarry length transition gy k := by
  induction k with
  | zero => exact zero
  | succ k ih => simp only [step, Reference.reverseCarry, ih]

theorem transitionSum_eq {α : Type} [Semiring α] (length : Nat) (transition initial : α)
    (saved gy acc carry : Nat → α)
    (carried : ∀ k,carry k=Reference.reverseCarry length transition gy k)
    (zero : acc 0=0)
    (step : ∀ k,acc (k+1)=acc k+(gy (length-1-k)+carry k)*
      (if length-1-k=0 then initial else saved (length-1-k-1))) (k : Nat) :
    acc k=Reference.transitionGradient length transition initial saved gy k := by
  induction k with
  | zero => exact zero
  | succ k ih => simp only [step, Reference.transitionGradient, Reference.previousState, carried, ih]

#axiomsClean state_eq
#axiomsClean carry_eq
#axiomsClean transitionSum_eq
end VeriTile.Bench.Repairs.DiagSsmTriton.HeadlineRecurrences
