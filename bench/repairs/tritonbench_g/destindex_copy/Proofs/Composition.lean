import bench.repairs.tritonbench_g.destindex_copy.Contract
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.PhysicalCorrect

namespace VeriTile.Bench.Repairs.DestindexCopy
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type}

def touched (m : Inputs) (w : Nat → Word) (s : Side) (x : Nat) : Prop :=
  DestindexCopyKv1.touched (config m s) (destination w) x

def Stage (m : Inputs) (w : Nat → Word) (s : Side) (before after : Nat → α) : Prop :=
  DestindexCopyKv1.RowCopy (config m s) (destination w)
    (fun x => before (addr (base m s .source) x)) (fun x => before (addr (base m s .output) x))
    (fun x => after (addr (base m s .output) x)) ∧
  (∀ cell, (¬∃ x, touched m w s x ∧ cell=addr (base m s .output) x) → after cell=before cell)

/-- Independent simultaneous initial-input copy for both caller output tensors. -/
def DualCopy (m : Inputs) (w : Nat → Word) (before after : Nat → α) : Prop := ∀ s,
  (∀ p a b, p<(config m s).count → a<(config m s).heads → b<(config m s).dims →
    after (addr (base m s .output) (flat (config m s) (destination w p) a b))=
      before (addr (base m s .source) (flat (config m s) p a b))) ∧
  (∀ x, x<((plane m s) .output).numel → ¬touched m w s x →
    after (addr (base m s .output) x)=before (addr (base m s .output) x))

def Outside (m : Inputs) (w : Nat → Word) (cell : Nat) : Prop :=
  ∀ s, ¬∃ x, touched m w s x ∧ cell=addr (base m s .output) x

theorem checkInputs_iff (m : Inputs) (w : Nat → Word) :
    checkInputs m w=.ok () ↔ InputContract m w := by
  have fields : InputContract m w ↔ DestindexCopyKv1.InputContract (plane m .nope) w ∧
      DestindexCopyKv1.InputContract (plane m .rope) w ∧ Separated m := by
    constructor
    · intro h; exact ⟨h.common .nope,h.common .rope,h.separated⟩
    · rintro ⟨a,b,c⟩; exact ⟨fun s => by cases s; exact a; exact b,c⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (m : Option Inputs) (w : Nat → Word) :
    checkArguments m w=.ok () ↔ Admitted m w := by
  cases m <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem preserves (m : Inputs) (w : Nat → Word) (h : InputContract m w) (s : Side)
    (before after : Nat → α) (stage : Stage m w s before after) (otherBase size : Nat)
    (separate : SpanSeparate (base m s .output) ((plane m s) .output).numel otherBase size)
    (x : Nat) (hx : x<size) : after (addr otherBase x)=before (addr otherBase x) := by
  apply stage.2
  rintro ⟨y,hy,he⟩
  obtain ⟨p,a,b,hp,ha,hb,rfl⟩ := hy
  have bound := DestindexCopyKv1.output_bound (plane m s) w (h.common s) p a b hp ha hb
  change flat (config m s) (destination w p) a b<((plane m s) .output).numel at bound
  unfold SpanSeparate at separate
  simp only [addr,DestindexCopyKv1.addr,flat,DestindexCopyKv1.flat] at he bound
  rcases separate with left|right <;> omega

theorem compose (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before middle after : Nat → α) (first : Stage m w .nope before middle)
    (second : Stage m w .rope middle after) :
    DualCopy m w before after ∧ (∀ cell, Outside m w cell → after cell=before cell) := by
  have keepNope := preserves m w h .rope middle after second _ _
    ((h.separated .rope .nope).2.2 (by decide))
  have keepRopeSource := preserves m w h .nope before middle first _ _ (h.separated .nope .rope).1
  have keepRopeOutput := preserves m w h .nope before middle first _ _
    ((h.separated .nope .rope).2.2 (by decide))
  constructor
  · intro s
    cases s with
    | nope =>
      constructor
      · intro p a b hp ha hb
        exact (keepNope _ (DestindexCopyKv1.output_bound (plane m .nope) w (h.common .nope) p a b hp ha hb)).trans
          (first.1.1 p a b hp ha hb)
      · intro x hx miss
        exact (keepNope x hx).trans (first.1.2 x miss)
    | rope =>
      constructor
      · intro p a b hp ha hb
        exact (second.1.1 p a b hp ha hb).trans
          (keepRopeSource _ (DestindexCopyKv1.source_bound (plane m .rope) w (h.common .rope) p a b hp ha hb))
      · intro x hx miss
        exact (second.1.2 x miss).trans (keepRopeOutput x hx)
  · intro cell outside
    rw [second.2 cell (outside .rope),first.2 cell (outside .nope)]

#axiomsClean checkArguments_iff
#axiomsClean compose
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopy
