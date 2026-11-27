/- Conditional scalar theories and successful kernel execution.
Numerical premises apply only to admitted scalar fragments in their declared
operand domain. Program signatures retain the input contract and IO windows. -/
import VeriTile.Triton.Float.GuardedStrict

namespace VeriTile.Triton.FP.Guarded
open Structural

/-- An interpretation of the fixed strict FP32 theory plus the displayed
experimental assumptions, observed at scalar `out`. The strict theory's
software identity is fixed in this type; this does not assert that its software
model satisfies the additional experimental equalities. Both executions
must exist, and both operand domains must hold, before a law can be used.

The theory is about FP32 values. Its laws are therefore required only on
states whose floating values are encoded FP32 words (`WordState`), and the
operand domain may contain only such words. A carrier can hold other
elements, such as a BF16/FP32 pair with disagreeing components; no kernel
reaches them, and a law about FP32 values says nothing about them. -/
def Models {α : Type} [Inhabited α] (R : Spec.Assumptions GuardedFragment)
    (M : Algebra α) (D : Domain α) : Prop :=
  (∀ kind x, D kind x → Word M x) ∧
  ∀ lhs rhs, @Spec.Derivation GuardedFragment fp32Theory R [lhs] [rhs] →
    ∀ s, WordState M s →
      TypedScalarDomain D lhs.dtype lhs.guards s → TypedScalarDomain D rhs.dtype rhs.guards s →
      ∀ a b, run (fragmentAlgebra lhs M) lhs.code s = some a →
        run (fragmentAlgebra rhs M) rhs.code s = some b →
        scalarOutput lhs.dtype a = scalarOutput rhs.dtype b

/-- Every guarded operand of a model is an encoded FP32 word. -/
theorem Models.word {α : Type} [Inhabited α] {R : Spec.Assumptions GuardedFragment}
    {M : Algebra α} {D : Domain α} (h : Models R M D) {kind : GuardKind} {x : α}
    (hx : D kind x) : Word M x := h.1 kind x hx

/-- The strict backend encodes each FP32 word as itself. -/
theorem strict_faithful (rest : Algebra (FP.Value .fp32)) :
    FaithfulWords (Software.strictFP32 rest) := fun _ _ h => h

/-- Each strict FP32 word is its own encoding in the strict backend. -/
theorem strict_word (rest : Algebra (FP.Value .fp32)) (x : FP.Value .fp32) :
    Word (Software.strictFP32 rest) x := ⟨x, rfl⟩

/-- A concrete witness for the strict theory with no experimental rules.
This does not assert that the same implementation models any relaxed atom. -/
theorem strict_models_empty (rest : Algebra (FP.Value .fp32)) :
    Models [] (Software.strictFP32 rest) strictDomain := by
  refine ⟨fun _ x _ => strict_word rest x, ?_⟩
  intro lhs rhs proof s _ hl _ a b ha hb
  have certificate := (List.forall₂_cons.mp (certified_of_empty proof)).1
  simpa only [ha, hb, Option.map_some, Option.some.injEq] using certificate.2 rest s hl

structure RegisterGuard where
  name : RegName
  shape : TileShape
  kind : GuardKind

/-- A domain contract checks intermediate operands of a common prefix. It is
part of the public signature; changing it is not an equivalence step. -/
structure Precondition where
  code : List ComputeStmt
  guards : List RegisterGuard

def Precondition.Holds {α : Type} [Inhabited α] (P : Precondition)
    (M : Algebra α) (D : Domain α) (s : State α) : Prop :=
  ∃ t, run M P.code s = some t ∧
    ∀ g ∈ P.guards, ∃ v, t.regs .real g.shape g.name = some v ∧ ∀ i, D g.kind (v i)

structure IO where
  io : KernelIO₁
  domain : Precondition
  defaultPrecision : Option ComputeDType := none

/-- Keep explicit precision tags; optionally supply the precision of ordinary
Triton expressions, including comparisons and `where` temporaries. -/
def IO.algebra {α : Type} (io : IO) (M : Algebra α) : Algebra α :=
  match io.defaultPrecision with
  | none => M
  | some p => M.withDefaultPrecision p

def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : IO) : Prop :=
  IO₁PrivateScratch lhs.io ∧ IO₁PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, lhs.domain.Holds (lhs.algebra M) D s →
      ∃ a b, Structural.exec (lhs.algebra M) lhs.io.kernel s = some a ∧
        Structural.exec (rhs.algebra M) rhs.io.kernel s = some b ∧
        (∀ i : Fin lhs.io.Bout,
          a.mem lhs.io.out (lhs.io.write (s.pids 0) + i.val) =
            b.mem rhs.io.out (rhs.io.write (s.pids 0) + i.val)) ∧
        IO₁Frame lhs.io s a ∧ IO₁Frame rhs.io s b

instance : Spec.ProgramSyntax IO where
  Statement := GuardedFragment
  theory := fp32Theory
  Signature := IO₁Signature × Precondition × Option ComputeDType
  signature p := (io₁Signature p.io, p.domain, p.defaultPrecision)
  body p := [⟨[], p.io.kernel.surfaceBody, .real, none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  -- These IO proofs use successful execution, never unchecked syntax rewrites.
  sameContext := fun lhs rhs => lhs = rhs

/-- A guarded one-input/two-output contract, such as mean and variance.
Its public FP specification uses the same equivalence notation as `IO`. -/
structure IO₁ₓ₂ where
  io : KernelIO₁ₓ₂
  domain : Precondition

def Equivalent₁ₓ₂ (R : Spec.Assumptions GuardedFragment) (lhs rhs : IO₁ₓ₂) : Prop :=
  IO₁ₓ₂PrivateScratch lhs.io ∧ IO₁ₓ₂PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, lhs.domain.Holds M D s →
      ∃ a b, Structural.exec M lhs.io.kernel s = some a ∧ Structural.exec M rhs.io.kernel s = some b ∧
        IO₁ₓ₂Outputs lhs.io rhs.io s a b ∧ IO₁ₓ₂Frame lhs.io s a ∧ IO₁ₓ₂Frame rhs.io s b

/-- Structural proofs remain usable under a guarded contract without adding
any numerical assumption. The specification separately checks signatures. -/
theorem Equivalent₁ₓ₂.ofStructural (R : Spec.Assumptions GuardedFragment) {lhs rhs : IO₁ₓ₂}
    (h : Structural.IO₁ₓ₂Equiv lhs.io rhs.io) : Equivalent₁ₓ₂ R lhs rhs :=
  ⟨h.1, h.2.1, fun α _ M _ _ s _ => h.2.2 α M s⟩

instance : Spec.ProgramSyntax IO₁ₓ₂ where
  Statement := GuardedFragment
  theory := fp32Theory
  Signature := IO₁ₓ₂Signature × Precondition
  signature p := (io₁ₓ₂Signature p.io, p.domain)
  body p := [⟨[], p.io.kernel.surfaceBody, .real, none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent₁ₓ₂
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.Guarded
