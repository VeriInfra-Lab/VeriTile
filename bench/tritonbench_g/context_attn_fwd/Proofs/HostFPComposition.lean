import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Launch.Grid
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostFPComposition
open VeriTile VeriTile.Triton FP.Structural FP.ProfiledRewrite FP.Guarded
noncomputable section

/-- Fresh registers, actual launch coordinates, shared current tensor memory. -/
def prepare {α : Type} (g : Grid) (idx : GridIndex g) (s : State α) : State α :=
  { s with regs := fun _ _ _ => none, pids := idx.toPids, numPids := g.toNumPids }

def schedule {α : Type} [Inhabited α] (M : Algebra α) (p : Program) (g : Grid) :
    List (GridIndex g) → State α → Option (State α)
  | [], s => some s
  | idx :: rest, s => (FP.Structural.exec (engine M) p.kernel (prepare g idx s)).bind
      (schedule M p g rest)

/-- Only actual reached operands are guarded. Failure is preserved; this
predicate contains no desired output equality or successful-execution premise. -/
def domain {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α) (p : Program) (g : Grid) :
    List (GridIndex g) → State α → Prop
  | [], _ => True
  | idx :: rest, s =>
      (∀ site ∈ p.domain, site.Holds M D (prepare g idx s)) ∧
      (∀ next, FP.Structural.exec (engine M) p.kernel (prepare g idx s) = some next →
        domain M D p g rest next)

/-- Recover the execution statement from the audited public derivation.
Signature preservation transports the exact reached-operand domain through
symmetry/transitivity; syntax-only steps retain the entire context. -/
theorem sound (R : Spec.Assumptions FP.GuardedFragment) (left right : Program)
    (h : Spec.FloatingPoint R left right) : Equivalent R left right := by
  have hd : Spec.ProgramDerivation (fun _ _ : Program => False) R left right := h.derivation
  clear h
  have result : (Spec.ProgramSyntax.signature left = Spec.ProgramSyntax.signature right) ∧
      Equivalent R left right := by
    induction hd with
    | refl p => exact ⟨rfl, by unfold FP.ProfiledRewrite.Equivalent; intros; rfl⟩
    | @«syntax» l r hs hc hp =>
      change l = r at hc
      subst r
      exact ⟨rfl, by unfold FP.ProfiledRewrite.Equivalent; intros; rfl⟩
    | structural hs hp => exact False.elim hp
    | numerical hs hp => exact ⟨hs, hp⟩
    | @symm l r deriv ih =>
      refine ⟨ih.1.symm, ?_⟩
      intro α inst M D hM st guards
      have domains : l.domain = r.domain := congrArg Prod.snd ih.1
      exact (ih.2 α M D hM st (by simpa only [domains] using guards)).symm
    | @trans l mid r first second ih1 ih2 =>
      refine ⟨ih1.1.trans ih2.1, ?_⟩
      intro α inst M D hM st guards
      have domains : l.domain = mid.domain := congrArg Prod.snd ih1.1
      exact (ih1.2 α M D hM st guards).trans
        (ih2.2 α M D hM st (by simpa only [← domains] using guards))
  exact result.2

theorem equivalent (R : Spec.Assumptions FP.GuardedFragment) (left right : Program)
    (heq : Equivalent R left right) {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (hM : Models R M D)
    (g : Grid) (jobs : List (GridIndex g)) (s : State α) (h : domain M D left g jobs s) :
    schedule M left g jobs s = schedule M right g jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    have he := heq α M D hM (prepare g idx s) h.1
    simp only [schedule]
    rw [← he]
    cases hx : FP.Structural.exec (engine M) left.kernel (prepare g idx s) with
    | none => rfl
    | some next => exact ih next (h.2 next hx)

def allPrograms (g : Grid) : List (GridIndex g) := by
  classical
  exact Finset.univ.toList

theorem allPrograms_complete (g : Grid) (idx : GridIndex g) : idx ∈ allPrograms g := by
  classical
  simp [allPrograms]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostFPComposition
