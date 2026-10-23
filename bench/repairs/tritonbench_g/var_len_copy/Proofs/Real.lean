import bench.repairs.tritonbench_g.var_len_copy.Proofs.Contract
import bench.repairs.tritonbench_g.var_len_copy.Proofs.Conformance

/-! Arbitrary payload interpretation of the same source-derived event trace.
Signed-int32 metadata and finite source controls are retained independently;
Real payloads do not pass through a finite-word decoder. -/
namespace VeriTile.Bench.Repairs.VarLenCopy.Mathematical
noncomputable section
variable {α : Type}
set_option maxHeartbeats 3000000

def copyEvent (s : Nat → α) (e : Event) : Nat → α := Function.update s e.1 (s e.2)
def execute (xs : List Event) (s : Nat → α) : Nat → α := xs.foldl copyEvent s

theorem execute_frame (xs : List Event) (s : Nat → α) (cell : Nat)
    (miss : ∀ e∈xs, e.1≠cell) : execute xs s cell=s cell := by
  induction xs generalizing s with
  | nil => rfl
  | cons e rest ih =>
    rw [execute,List.foldl_cons]
    change execute rest (copyEvent s e) cell=s cell
    rw [ih _ (fun f hf => miss f (List.mem_cons_of_mem _ hf))]
    simp only [copyEvent,Function.update_of_ne (Ne.symm (miss e (by simp)))]

def snapshot (xs : List Event) (initial s : Nat → α) : Nat → α :=
  xs.foldl (fun t e => Function.update t e.1 (initial e.2)) s

theorem execute_snapshot (xs : List Event) (initial s : Nat → α)
    (reads : ∀ e∈xs, s e.2=initial e.2)
    (separate : ∀ e∈xs, ∀ f∈xs, e.1≠f.2) : execute xs s=snapshot xs initial s := by
  induction xs generalizing s with
  | nil => rfl
  | cons e rest ih =>
    have read := reads e (by simp)
    simp only [execute,List.foldl_cons,snapshot,copyEvent,read]
    change execute rest (Function.update s e.1 (initial e.2))=
      snapshot rest initial (Function.update s e.1 (initial e.2))
    apply ih
    · intro f hf
      rw [Function.update_of_ne (Ne.symm (separate e (by simp) f (by simp [hf])))]
      exact reads f (by simp [hf])
    · intro f hf g hg
      exact separate f (by simp [hf]) g (by simp [hg])

theorem snapshot_read (xs : List Event) (initial s : Nat → α) (cell : Nat) (v : α)
    (writers : ∀ e∈xs, e.1=cell → initial e.2=v)
    (present : (∃ e∈xs, e.1=cell) ∨ s cell=v) : snapshot xs initial s cell=v := by
  induction xs generalizing s with
  | nil => simpa [snapshot] using present
  | cons e rest ih =>
    simp only [snapshot,List.foldl_cons]
    change snapshot rest initial (Function.update s e.1 (initial e.2)) cell=v
    apply ih _ (fun f hf => writers f (by simp [hf]))
    by_cases he : e.1=cell
    · apply Or.inr
      rw [he,Function.update_self]
      exact writers e (by simp) he
    · rcases present with ⟨f,hf,hfc⟩ | hs
      · apply Or.inl
        refine ⟨f,?_,hfc⟩
        rcases List.mem_cons.mp hf with h | h
        · subst f; exact False.elim (he hfc)
        · exact h
      · exact Or.inr (by rwa [Function.update_of_ne (Ne.symm he)])


def sourceEvents (l : Layout) (metadata : Memory) : List Event :=
  (List.range l.count).flatMap fun p => Native.sourceEvents l (loadSegment l metadata p)

def run (l : Layout) (metadata : Memory) (initial : Nat → α) : Nat → α :=
  execute (sourceEvents l metadata) initial

theorem sourceEvents_exact (l : Layout) (r : Request) (metadata : Memory)
    (prep : Prepared l r metadata) (valid : Valid r) :
    sourceEvents l metadata = allEvents l r (List.range l.count) := by
  apply List.flatMap_congr
  intro p hp
  have bound : p < r.count := by simpa [prep.counts] using hp
  rw [metadata_load l r metadata prep p bound,Native.source_events l r metadata prep valid p bound]

theorem correct (l : Layout) (r : Request) (metadata : Memory)
    (prep : Prepared l r metadata) (valid : Valid r) (initial : Nat → α) :
    SegmentCopy r (fun i => initial (addr l.source i)) (fun i => initial (addr l.output i))
      (fun i => run l metadata initial (addr l.output i)) ∧
    (∀ cell, (¬∃ x, Touched r x ∧ cell = addr l.output x) → run l metadata initial cell = initial cell) := by
  unfold run
  rw [sourceEvents_exact l r metadata prep valid]
  have bound : ∀ p ∈ List.range l.count, p < r.count := by
    intro p hp; simpa [prep.counts] using hp
  have complete : ∀ p, p < r.count → p ∈ List.range l.count := by
    intro p hp; simpa [prep.counts] using hp
  have mem := allEvents_mem l r metadata prep valid (List.range l.count) bound
  have separate : ∀ e ∈ allEvents l r (List.range l.count),
      ∀ f ∈ allEvents l r (List.range l.count), e.1 ≠ f.2 := by
    intro e he f hf
    obtain ⟨p,i,hp,hi,rfl⟩ := (mem e).mp he
    obtain ⟨q,j,hq,hj,rfl⟩ := (mem f).mp hf
    have ho := (valid.outputBound p (bound p hp) (by omega)).2
    have hs := (valid.sourceBound q (bound q hq) (by omega)).2
    exact (prep.disjoint _ (by rw [prep.outputSize]; omega)).1 _ (by rw [prep.sourceSize]; omega)
  have frame : ∀ cell, (¬∃ x, Touched r x ∧ cell=addr l.output x) →
      execute (allEvents l r (List.range l.count)) initial cell=initial cell := by
    intro cell miss
    apply execute_frame
    intro e he heq
    obtain ⟨p,i,hp,hi,rfl⟩ := (mem e).mp he
    exact miss ⟨(r.segment p).dest+i,⟨p,i,bound p hp,hi,rfl⟩,heq.symm⟩
  refine ⟨⟨?_,?_⟩,frame⟩
  · intro p i hp hi
    rw [execute_snapshot _ initial initial (fun _ _ => rfl) separate]
    apply snapshot_read
    · intro e he heq
      obtain ⟨q,j,hq,hj,rfl⟩ := (mem e).mp he
      have same : (r.segment q).dest+j=(r.segment p).dest+i := by
        change addr l.output ((r.segment q).dest+j)=addr l.output ((r.segment p).dest+i) at heq
        unfold addr at heq; omega
      obtain ⟨rfl,rfl⟩ := valid.unique q p j i (bound q hq) hp hj hi same
      rfl
    · exact Or.inl ⟨_,(mem _).mpr ⟨p,i,complete p hp,hi,rfl⟩,rfl⟩
  · intro x _ hx
    apply frame
    rintro ⟨y,hy,he⟩
    have equal : x=y := by unfold addr at he; omega
    exact hx (equal ▸ hy)

#axiomsClean sourceEvents_exact
#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.Mathematical
