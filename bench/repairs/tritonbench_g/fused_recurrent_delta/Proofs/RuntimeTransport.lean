import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IOResult
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Task-local structural bridge for a preserved runtime register. It checks
actual statement execution and trace safety, including each loop successor.
It supplies no floating arithmetic identity. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeTransport
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def Bound (value : α) (s : State α) : Prop :=
  s.regs .real [] "scale" = some (fun _ => value)

structure Atom (M : Algebra α) (value : α) (a b : ComputeStmt) : Prop where
  executes : ∀ s, Bound value s → step M a s = step M b s
  safe : ∀ bounds s, Bound value s → (Safety.stepSafe M bounds a s ↔ Safety.stepSafe M bounds b s)
  keeps : ∀ s t, Bound value s → step M a s = some t → Bound value t

structure Pair (M : Algebra α) (value : α) (a b : List ComputeStmt) : Prop where
  executes : ∀ s, Bound value s → run M a s = run M b s
  safe : ∀ bounds s, Bound value s → (Safety.traceSafeList M bounds a s ↔ Safety.traceSafeList M bounds b s)
  keeps : ∀ s t, Bound value s → run M a s = some t → Bound value t

 theorem nil (M : Algebra α) (value : α) : Pair M value [] [] := by
  constructor
  · intros; rfl
  · intros; rfl
  · intro s t hs h; simp only [run,Option.some.injEq] at h; subst t; exact hs

 theorem cons {M : Algebra α} {value : α} {a b : ComputeStmt} {as bs : List ComputeStmt}
    (head : Atom M value a b) (tail : Pair M value as bs) : Pair M value (a::as) (b::bs) := by
  constructor
  · intro s hs
    simp only [run,←head.executes s hs]
    cases he : step M a s with
    | none => rfl
    | some t => exact tail.executes t (head.keeps s t hs he)
  · intro bounds s hs
    simp only [Safety.traceSafeList,←head.executes s hs,head.safe bounds s hs]
    cases he : step M a s with
    | none => rfl
    | some t => exact and_congr Iff.rfl (tail.safe bounds t (head.keeps s t hs he))
  · intro s t hs h
    simp only [run] at h
    cases he : step M a s with
    | none => simp [he] at h
    | some u => exact tail.keeps u t (head.keeps s u hs he) (by simpa [he] using h)

 theorem fold_regs {ι : Type} (l : List ι) (active : ι → Bool)
    (address : ι → RegionName × Nat) (value : ι → Cell α) (s : State α) :
    (l.foldl (fun t i => if active i then t.write (address i).1 (address i).2 (value i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => simp only [List.foldl_cons,ih]; split <;> rfl

 theorem fold_preserves {ι : Type} (l : List ι) (f : State α → ι → State α)
    (hf : ∀ s i, (f s i).regs=s.regs) (s : State α) : (l.foldl f s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => exact (ih (f s i)).trans (hf s i)

 theorem store_regs (M : Algebra α) (mem : MemAccess d sh) (mask : MaskOpt d sh)
    (values : Values α d sh) (s t : State α) (h : FP.Structural.store M mem mask values s=some t) : t.regs=s.regs := by
  cases mem <;> cases mask
  all_goals
    simp only [FP.Structural.store,bind,pure,Option.bind,Option.map] at h
    repeat' first | (simp only [bind,pure,Option.bind,Option.map] at h) | split at h | contradiction
    all_goals cases h
    all_goals apply fold_preserves
    all_goals intro u i; first | rfl | (split <;> rfl)

 theorem assignment (M : Algebra α) (value : α) (name : RegName) (hn : name ≠ "scale")
    (a b : ComputeExpr d sh)
    (same : ∀ s, Bound value s → evalExpr M a s=evalExpr M b s)
    (safe : ∀ bounds s, Bound value s → (Safety.safeExpr M bounds s a ↔ Safety.safeExpr M bounds s b)) :
    Atom M value (.assign d sh name a) (.assign d sh name b) := by
  constructor
  · intro s hs; simp only [step,same s hs]
  · intro bounds s hs; simpa only [Safety.stepSafe] using safe bounds s hs
  · intro s t hs h
    cases he : evalExpr M a s with
    | none => simp [step,he] at h
    | some v =>
      have ht : s.setReg name d sh v=t := by simpa [step,he] using h
      subst t
      simpa [Bound,State.setReg,hn,Ne.symm hn] using hs

 theorem sameAssignment (M : Algebra α) (value : α) (name : RegName) (hn : name ≠ "scale")
    (a : ComputeExpr d sh) : Atom M value (.assign d sh name a) (.assign d sh name a) :=
  assignment M value name hn a a (by intros; rfl) (by intros; rfl)

 theorem sameStore (M : Algebra α) (value : α) (mem : MemAccess d sh)
    (a : ComputeExpr d sh) (mask : MaskOpt d sh) :
    Atom M value (.store d sh mem a mask) (.store d sh mem a mask) := by
  constructor
  · intros; rfl
  · intros; rfl
  · intro s t hs h
    cases he : evalExpr M a s with
    | none => simp [step,he] at h
    | some v =>
      have ht : FP.Structural.store M mem mask v s=some t := by simpa [step,he] using h
      unfold Bound
      rw [store_regs M mem mask v s t ht]
      exact hs

 theorem barrier (M : Algebra α) (value : α) :
    Atom M value (.effectMarker "tl.debug_barrier") (.effectMarker "tl.debug_barrier") := by
  constructor
  · intros; rfl
  · intros; rfl
  · intro s t hs h; simp only [step,Option.some.injEq] at h; subst t; exact hs

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeTransport
