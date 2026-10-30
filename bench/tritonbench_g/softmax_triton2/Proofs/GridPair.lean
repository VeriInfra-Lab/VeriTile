import bench.tritonbench_g.softmax_triton2.Proofs.Pair

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open VeriTile.Triton FP.Structural FP.Guarded
open scoped VeriTile.Spec
noncomputable section

/-- Actual whole-kernel launches in the stated row order. Every row starts
with fresh registers and consumes the actual preceding memory. Only the final
memory is observed; all typed exterior cells remain part of this observation. -/
def gridExec {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel)
    (base : State α) : List Nat → (RegionName → Nat → Cell α) →
      Option (RegionName → Nat → Cell α)
  | [], mem => some mem
  | p :: rest, mem => do
      let t ← exec M kernel { base with mem := mem, regs := (fun _ _ _ => none), pids := (fun ax => if ax = 0 then p else 0) }
      gridExec M kernel base rest t.mem

structure GridProgram where
  kernel : ComputeKernel
  programs : List Nat

def GridEquivalent (R : Spec.Assumptions FP.GuardedFragment)
    (lhs rhs : GridProgram) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    Ports M D → ∀ base mem, ∃ final,
      gridExec (FP.ProfiledRewrite.engine M) lhs.kernel base lhs.programs mem = some final ∧
      gridExec (FP.ProfiledRewrite.engine M) rhs.kernel base rhs.programs mem = some final

instance : Spec.ProgramSyntax GridProgram where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := (List RegionName × List RegionName) × List Nat
  signature p := (match p.kernel with | .mk inputs outputs _ => (inputs,outputs),p.programs)
  body p := [⟨[],p.kernel.surfaceBody,.real,none⟩]
  structural := some (fun _ _ => False)
  numerical := GridEquivalent
  sameContext := fun a b => a = b

/-- Successful complete-row equivalence composes through every actual launch,
including overlapping windows and arbitrary row orders. No input restriction,
replacement formula, source evaluator assumption or numerical grid law is used. -/
theorem grid_of_row (R : Spec.Assumptions FP.GuardedFragment)
    (lhs rhs : Program) (rows : List Nat) (h : Equivalent R lhs rhs) :
    GridEquivalent R ⟨lhs.kernel,rows⟩ ⟨rhs.kernel,rows⟩ := by
  intro α _ M D model ports base mem
  induction rows generalizing mem with
  | nil => exact ⟨mem,rfl,rfl⟩
  | cons p rest ih =>
    let launch : State α := { base with mem := mem, regs := (fun _ _ _ => none), pids := (fun ax => if ax = 0 then p else 0) }
    obtain ⟨a,b,ha,hb,same⟩ := h α M D model ports launch
    dsimp [launch] at ha hb
    obtain ⟨final,hl,hr⟩ := ih a.mem
    refine ⟨final,?_,?_⟩
    · simpa only [gridExec,ha,Option.bind_some] using hl
    · simp only [gridExec,hb,Option.bind_some]
      change gridExec (FP.ProfiledRewrite.engine M) rhs.kernel base rest b.mem = some final
      rw [←same]
      exact hr

#axiomsClean grid_of_row
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
