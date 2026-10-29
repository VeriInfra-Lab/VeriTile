import VeriTile.Triton.Float.GuardedProgramIO

/-! Regression for conditional model premises escaping through transitivity.
The invariant covers every derivation, including arbitrary intermediate
backends and rule sets. A successful empty schedule cannot equal a failing one.
-/
namespace VeriTile.Triton.FP.Structural.GuardedProgramRegression
open ProgramIO GuardedProgram
open scoped VeriTile.Spec
noncomputable section

section Invariant
variable {α : Type} [Inhabited α] {io : ProgramIO α α}
  {allowed : io.Inputs → Prop} {domain : FP.Guarded.Domain α}
  {p q r : GuardedProgram io allowed domain}

private theorem equivalent_symm (h : Equivalent p q) : Equivalent q p := by
  intro vals ha bounds hb initial hi
  obtain ⟨sp, sq, l, r, hl, hr, outputs, fl, fr⟩ := h vals ha bounds hb initial hi
  exact ⟨sq, sp, r, l, hr, hl, fun o j active => (outputs o j active).symm, fr, fl⟩

private theorem equivalent_trans (hpq : Equivalent p q) (hqr : Equivalent q r) :
    Equivalent p r := by
  intro vals ha bounds hb initial hi
  obtain ⟨sp, _, l, m, hl, hm, lm, fl, _⟩ := hpq vals ha bounds hb initial hi
  obtain ⟨_, sr, m', n, hm', hn, mn, _, fn⟩ := hqr vals ha bounds hb initial hi
  have same : m = m' := Option.some.inj (hm.symm.trans hm')
  subst m'
  exact ⟨sp, sr, l, n, hl, hn, fun o j active =>
    (lm o j active).trans (mn o j active), fl, fn⟩

/-- A non-identity derivation retains successful executions, safety, equal
typed outputs and both frames, regardless of its intermediate programs. -/
theorem derivation_preserves_execution (R : Spec.Assumptions FP.GuardedFragment)
    (h : p ≡[R] q) : p = q ∨ Equivalent p q := by
  have hd : Spec.ProgramDerivation Equivalent R p q := h.derivation
  clear h
  induction hd with
  | refl _ => exact Or.inl rfl
  | «syntax» _ context _ => exact Or.inl context
  | structural _ proof => exact Or.inr proof
  | numerical _ proof => exact Or.inr proof.2.2
  | symm _ ih =>
    rcases ih with same | eqv
    · exact Or.inl same.symm
    · exact Or.inr (equivalent_symm eqv)
  | trans _ _ left right =>
    rcases left with same | eqv
    · subst same; exact right
    · rcases right with same | eqv'
      · subst same; exact Or.inr eqv
      · exact Or.inr (equivalent_trans eqv eqv')

theorem nonmodel_cannot_be_numerical_intermediate
    (R : Spec.Assumptions FP.GuardedFragment) (hbad : ¬ q.Models R) :
    ¬ Numerical R p q ∧ ¬ Numerical R q r := by
  exact ⟨fun h => hbad h.2.1, fun h => hbad h.1⟩

/-- Supplying genuine certificates and an execution proof still permits a
numerical step, symmetry and transitivity. -/
theorem certified_composition (R : Spec.Assumptions FP.GuardedFragment)
    (hpq : p.outputPrecision = q.outputPrecision)
    (hrq : r.outputPrecision = q.outputPrecision)
    (mp : p.Models R) (mq : q.Models R) (mr : r.Models R)
    (pq : Equivalent p q) (rq : Equivalent r q) : p ≡[R] r :=
  (of_numerical p q hpq R mp mq pq).trans
    (of_numerical r q hrq R mr mq rq).symm
end Invariant

def emptyIO : ProgramIO Nat Nat where
  inputs := []
  outputs := []
  scratch := []
  buffers := []
  read := fun i => Fin.elim0 i
  inputMask := fun i => Fin.elim0 i
  write := fun i => Fin.elim0 i
  outputMask := fun i => Fin.elim0 i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => []
  encode := id
  observe := fun i => Fin.elim0 i

abbrev Program := GuardedProgram emptyIO (fun _ => True) (fun _ _ => True)

def idle : Program := ⟨[], []⟩

def failingJob (M : Algebra Nat) : Invocation Nat where
  kernel := .mk [] [] [.assign .nat [] "v" (.alg (.ref .nat [] "missing"))]
  backend := M
  clearRegisters := true
  pids := fun _ => 0

def failing (M : Algebra Nat) : Program := ⟨[failingJob M], []⟩

def initial : State Nat where
  mem := fun _ _ => .mk .nat 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem successful_vs_failed_not_equivalent
    (R : Spec.Assumptions FP.GuardedFragment) (M : Algebra Nat) :
    ¬ (idle ≡[R] failing M) := by
  intro h
  rcases derivation_preserves_execution R h with same | eqv
  · have jobs := congrArg GuardedProgram.jobs same
    simp [idle, failing] at jobs
  · let vals : emptyIO.Inputs := fun i => Fin.elim0 i
    have hb : emptyIO.InBounds (fun _ => 0) vals := by
      refine ⟨fun _ _ => rfl, ?_, ?_, ?_⟩ <;> intro i <;> exact Fin.elim0 i
    have hi : emptyIO.InputCells vals initial := by intro i; exact Fin.elim0 i
    obtain ⟨_, _, left, right, _, failedRun, _⟩ :=
      eqv vals trivial (fun _ => 0) hb initial hi
    simp [implementation, Implementation.toProgramIO, Implementation.toVariant,
      Implementation.relativeVariant, Variant.toProgramIO, failing, failingJob,
      Invocation.run, Invocation.initial, bindRegisters, exec, run,
      step, evalExpr, evalOp_unfold] at failedRun

#axiomsClean derivation_preserves_execution
#axiomsClean nonmodel_cannot_be_numerical_intermediate
#axiomsClean certified_composition
#axiomsClean successful_vs_failed_not_equivalent
end
end VeriTile.Triton.FP.Structural.GuardedProgramRegression
