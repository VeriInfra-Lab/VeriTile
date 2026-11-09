import VeriTile.Triton.Float.IntegerDot
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedAccumulatorCoordinates
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def context (plans : (k row col split iteration : Nat) → IntegerDot.Plan k) : IntegerDot.Context where
  rows := 0
  cols := 0
  rowRegister := "row"
  columnRegister := "column"
  iterationRegister := "k"
  splitAxis := 1
  plans := plans
  mode := .accumulatorNat
  signedAccumulator := Bool.true

def constant (x : Int) : Op .int [1,1] := .full [1,1] (.constInt x)

def signedState (s : State α) : State α :=
  { s with regs := fun d sh name => match d,sh with
      | .int,[] => some (fun _=>if name="row" then 2 else if name="column" then 3 else 4)
      | .nat,[] => some (fun _=>if name="row" then 99 else if name="column" then 98 else 97)
      | d,sh => s.regs d sh name }

/-- Stale Nat slots cannot select different instructions for signed source
coordinates. All three coordinates and the incoming carry remain observable. -/
theorem signed_slots_ignore_stale_nat {α : Type} [Inhabited α] (M : Algebra α)
    (a : IntegerDot.Arithmetic) (plans : (k row col split iteration : Nat) → IntegerDot.Plan k)
    (s : State α) :
    IntegerDot.accumulateNat M a (context plans) (constant 2) (constant 3) (constant 5) (signedState s) =
      some (fun i => (plans 1 (2+i.1.val) (3+i.2.1.val) (s.pids 1) 4).evalInt a
        (fun _=>BitVec.ofInt 8 2) (fun _=>BitVec.ofInt 8 3) 5) := by
  simp [IntegerDot.accumulateNat,context,constant,signedState,evalOp_unfold,State.setReg]

/-- Existing Nat profiles retain their original coordinate slots. -/
theorem default_profile_keeps_nat_slots {α : Type} [Inhabited α] (M : Algebra α)
    (a : IntegerDot.Arithmetic) (plans : (k row col split iteration : Nat) → IntegerDot.Plan k)
    (s : State α) :
    IntegerDot.accumulateNat M a { context plans with signedAccumulator := Bool.false }
      (constant 2) (constant 3) (constant 5) (signedState s) =
      some (fun i => (plans 1 (99+i.1.val) (98+i.2.1.val) (s.pids 1) 97).evalInt a
        (fun _=>BitVec.ofInt 8 2) (fun _=>BitVec.ofInt 8 3) 5) := by
  simp [IntegerDot.accumulateNat,context,constant,signedState,evalOp_unfold]

/-- Missing signed registers retain ordinary failure even when Nat registers
with the same names are present. No default coordinate is substituted. -/
theorem missing_signed_slot_fails {α : Type} [Inhabited α] (M : Algebra α)
    (a : IntegerDot.Arithmetic) (plans : (k row col split iteration : Nat) → IntegerDot.Plan k)
    (s : State α) (missing : s.regs .int [] "row"=none) :
    IntegerDot.accumulateNat M a (context plans) (constant 2) (constant 3) (constant 5)
      (s.setReg "row" .nat [] (fun _=>99)) = none := by
  simp [IntegerDot.accumulateNat,context,constant,evalOp_unfold,State.setReg,missing]

#axiomsClean default_profile_keeps_nat_slots
#axiomsClean signed_slots_ignore_stale_nat
#axiomsClean missing_signed_slot_fails
end
end VeriTile.Tests.SignedAccumulatorCoordinates
