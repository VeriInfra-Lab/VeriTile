import VeriTile.Triton.Float.GuardedSemantics
import VeriTile.Triton.Float.StrictBackend
import Mathlib.Data.List.Forall2

/-! Strict certificates are actual executions in one named software profile.
They are independent of the experimental rule list. A model of the generated
mixed theory is still an interpretation of that theory, not a claim that all
its empirical rewrites are exact identities of this software implementation. -/
namespace VeriTile.Triton.FP.Guarded
open Structural

def strictDomain : Domain (FP.Value .fp32)
  | .finite, x => x.toRat?.isSome = true
  | .nonzero, x => ∃ q, x.toRat? = some q ∧ q ≠ 0
  | .positive, x => ∃ q, x.toRat? = some q ∧ 0 < q

/-- Preserve guards, channel and default precision, then compare actual scalar
execution observations, including failure. Only the covered FP32 primitives
are fixed: all other operations remain arbitrary in `rest`, so an unsupported
operation cannot acquire a numerical law merely from sharing a carrier type. -/
def StrictFragmentSound (lhs rhs : GuardedFragment) : Prop :=
  (lhs.dtype, lhs.guards, lhs.defaultPrecision) =
    (rhs.dtype, rhs.guards, rhs.defaultPrecision) ∧
  ∀ (rest : Algebra (FP.Value .fp32)) (s : State (FP.Value .fp32)),
    TypedScalarDomain strictDomain lhs.dtype lhs.guards s →
    (run (fragmentAlgebra lhs (Software.strictFP32 rest)) lhs.code s).map
        (fun t => scalarOutput lhs.dtype t) =
      (run (fragmentAlgebra rhs (Software.strictFP32 rest)) rhs.code s).map
        (fun t => scalarOutput rhs.dtype t)

theorem StrictFragmentSound.refl (f : GuardedFragment) : StrictFragmentSound f f :=
  ⟨rfl, fun _ _ _ => rfl⟩

theorem StrictFragmentSound.symm {lhs rhs : GuardedFragment}
    (h : StrictFragmentSound lhs rhs) : StrictFragmentSound rhs lhs := by
  refine ⟨h.1.symm, fun rest s hd => ?_⟩
  have dtype : lhs.dtype = rhs.dtype := congrArg Prod.fst h.1
  have guards : lhs.guards = rhs.guards := congrArg (fun p => p.2.1) h.1
  exact (h.2 rest s (by simpa only [dtype, guards] using hd)).symm

theorem StrictFragmentSound.trans {lhs middle rhs : GuardedFragment}
    (h₁ : StrictFragmentSound lhs middle) (h₂ : StrictFragmentSound middle rhs) :
    StrictFragmentSound lhs rhs := by
  refine ⟨h₁.1.trans h₂.1, fun rest s hd => ?_⟩
  have dtype : lhs.dtype = middle.dtype := congrArg Prod.fst h₁.1
  have guards : lhs.guards = middle.guards := congrArg (fun p => p.2.1) h₁.1
  exact (h₁.2 rest s hd).trans (h₂.2 rest s (by simpa only [← dtype, ← guards] using hd))

/-- This named instance is fixed in every guarded public program view and in
`Models`; public equivalence statements cannot substitute another theory. -/
instance fp32Theory : Spec.StatementTheory GuardedFragment where
  modelID := "VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values"
  Certified := List.Forall₂ StrictFragmentSound

theorem strict_derivation (R : Spec.Assumptions GuardedFragment)
    {lhs rhs : GuardedFragment} (certificate : StrictFragmentSound lhs rhs) :
    @Spec.Derivation GuardedFragment fp32Theory R [lhs] [rhs] :=
  .strict (.cons certificate .nil)

/-- With no empirical atoms, every derived list equality still has an actual
execution certificate in the fixed software profile. -/
theorem certified_of_empty {lhs rhs : List GuardedFragment}
    (h : @Spec.Derivation GuardedFragment fp32Theory [] lhs rhs) :
    List.Forall₂ StrictFragmentSound lhs rhs := by
  have reflexive (xs : List GuardedFragment) : List.Forall₂ StrictFragmentSound xs xs := by
    induction xs with
    | nil => exact .nil
    | cons x xs ih => exact .cons (StrictFragmentSound.refl x) ih
  have chain {xs ys zs : List GuardedFragment}
      (h₁ : List.Forall₂ StrictFragmentSound xs ys)
      (h₂ : List.Forall₂ StrictFragmentSound ys zs) :
      List.Forall₂ StrictFragmentSound xs zs := by
    induction h₁ generalizing zs with
    | nil => exact h₂
    | cons head _ ih =>
      cases h₂ with
      | cons head' tail => exact .cons (head.trans head') (ih tail)
  induction h with
  | refl xs => exact reflexive xs
  | atom e he _ => simp at he
  | symm _ ih => exact ih.flip.imp (fun _ _ sound => StrictFragmentSound.symm sound)
  | trans _ _ ih₁ ih₂ => exact chain ih₁ ih₂
  | frame pre post _ ih =>
    exact List.rel_append (List.rel_append (reflexive pre) ih) (reflexive post)
  | strict certificate => exact certificate

end VeriTile.Triton.FP.Guarded
