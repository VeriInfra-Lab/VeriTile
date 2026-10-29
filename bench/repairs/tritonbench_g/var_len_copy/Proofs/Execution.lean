import bench.repairs.tritonbench_g.var_len_copy.Proofs.Index64

namespace VeriTile.Bench.Repairs.VarLenCopy
noncomputable section
set_option maxHeartbeats 3000000

theorem execute_frame (xs : List Event) (s : Memory) (cell : Nat)
    (miss : ∀ e∈xs, e.1≠cell) : execute xs s cell=s cell := by
  induction xs generalizing s with
  | nil => rfl
  | cons e rest ih =>
    rw [execute,List.foldl_cons]
    change execute rest (copyEvent s e) cell=s cell
    rw [ih _ (fun f hf => miss f (List.mem_cons_of_mem _ hf))]
    simp only [copyEvent,Function.update_of_ne (Ne.symm (miss e (by simp)))]

def snapshot (xs : List Event) (initial s : Memory) : Memory :=
  xs.foldl (fun t e => Function.update t e.1 (initial e.2)) s

theorem execute_snapshot (xs : List Event) (initial s : Memory)
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

theorem snapshot_read (xs : List Event) (initial s : Memory) (cell : Nat) (v : Word)
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

def allEvents (l : Layout) (r : Request) (order : List Nat) : List Event :=
  order.flatMap (fun p => events l (r.segment p))

theorem allEvents_mem (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) (order : List Nat)
    (bound : ∀ p∈order,p<r.count) (e : Event) :
    e∈allEvents l r order ↔ ∃ p i, p∈order ∧ i<(r.segment p).len ∧
      e=(addr l.output ((r.segment p).dest+i),addr l.source ((r.segment p).old+i)) := by
  simp only [allEvents,List.mem_flatMap]
  constructor
  · rintro ⟨p,hp,he⟩
    obtain ⟨i,hi,he⟩ := (events_mem l r s prep valid p (bound p hp) e).mp he
    exact ⟨p,i,hp,hi,he⟩
  · rintro ⟨p,i,hp,hi,he⟩
    exact ⟨p,hp,(events_mem l r s prep valid p (bound p hp) e).mpr ⟨i,hi,he⟩⟩

theorem events_frame (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) (p : Nat) (hp : p<r.count)
    (cell : Nat) (miss : ∀ i, i<l.outputSize → addr l.output i≠cell) :
    execute (events l (r.segment p)) s cell=s cell := by
  apply execute_frame
  intro e he
  obtain ⟨i,hi,rfl⟩ := (events_mem l r s prep valid p hp e).mp he
  apply miss
  have h := (valid.outputBound p hp (by omega)).2
  rw [prep.outputSize]
  omega

theorem prepared_execute (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) (p : Nat) (hp : p<r.count) :
    Prepared l r (execute (events l (r.segment p)) s) := by
  refine {prep with oldCells:=?_,lengthCells:=?_,newCells:=?_}
  · intro q hq
    rw [events_frame l r s prep valid p hp _ (fun i hi => ((prep.disjoint i hi).2 q hq).1)]
    exact prep.oldCells q hq
  · intro q hq
    rw [events_frame l r s prep valid p hp _ (fun i hi => ((prep.disjoint i hi).2 q hq).2.1)]
    exact prep.lengthCells q hq
  · intro q hq
    rw [events_frame l r s prep valid p hp _ (fun i hi => ((prep.disjoint i hi).2 q hq).2.2)]
    exact prep.newCells q hq

theorem launch_events (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) (order : List Nat)
    (bound : ∀ p∈order,p<r.count) :
    order.foldl (fun t p => program l p t) s=execute (allEvents l r order) s := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
    have hp := bound p (by simp)
    have hr := fun q hq => bound q (List.mem_cons_of_mem p hq)
    rw [List.foldl_cons]
    have step : program l p s=execute (events l (r.segment p)) s := by
      rw [program,metadata_load l r s prep p hp]
    rw [step]
    rw [ih _ (prepared_execute l r s prep valid p hp) hr]
    simp only [allEvents,List.flatMap_cons,execute,List.foldl_append]

/-- Actual finite native event execution, including all metadata prologues.
Every program order with complete coverage has the same initial-input outputs. -/
theorem ordered_correct (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) (order : List Nat)
    (bound : ∀ p∈order,p<r.count) (complete : ∀ p,p<r.count → p∈order) :
    (∀ p i,p<r.count → i<(r.segment p).len →
      (order.foldl (fun t q => program l q t) s) (addr l.output ((r.segment p).dest+i))=
        s (addr l.source ((r.segment p).old+i))) ∧
    (∀ cell,(¬∃ x,Touched r x ∧ cell=addr l.output x) →
      (order.foldl (fun t q => program l q t) s) cell=s cell) := by
  rw [launch_events l r s prep valid order bound]
  have mem := allEvents_mem l r s prep valid order bound
  have separate : ∀ e∈allEvents l r order,∀ f∈allEvents l r order,e.1≠f.2 := by
    intro e he f hf
    obtain ⟨p,i,hp,hi,rfl⟩ := (mem e).mp he
    obtain ⟨q,j,hq,hj,rfl⟩ := (mem f).mp hf
    have ho := (valid.outputBound p (bound p hp) (by omega)).2
    have hs := (valid.sourceBound q (bound q hq) (by omega)).2
    exact (prep.disjoint _ (by rw [prep.outputSize];omega)).1 _ (by rw [prep.sourceSize];omega)
  constructor
  · intro p i hp hi
    rw [execute_snapshot _ s s (fun _ _ => rfl) separate]
    apply snapshot_read
    · intro e he heq
      obtain ⟨q,j,hq,hj,rfl⟩ := (mem e).mp he
      have same : (r.segment q).dest+j=(r.segment p).dest+i := by
        change addr l.output ((r.segment q).dest+j)=addr l.output ((r.segment p).dest+i) at heq
        unfold addr at heq;omega
      obtain ⟨rfl,rfl⟩ := valid.unique q p j i (bound q hq) hp hj hi same
      rfl
    · apply Or.inl
      refine ⟨(addr l.output ((r.segment p).dest+i),addr l.source ((r.segment p).old+i)),?_,rfl⟩
      exact (mem _).mpr ⟨p,i,complete p hp,hi,rfl⟩
  · intro cell miss
    apply execute_frame
    intro e he heq
    obtain ⟨p,i,hp,hi,rfl⟩ := (mem e).mp he
    exact miss ⟨(r.segment p).dest+i,⟨p,i,bound p hp,hi,rfl⟩,heq.symm⟩

theorem launch_correct (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) :
    SegmentCopy r (fun i=>s (addr l.source i)) (fun i=>s (addr l.output i))
      (fun i=>launch l s (addr l.output i)) ∧
    (∀ cell,(¬∃ x,Touched r x ∧ cell=addr l.output x) → launch l s cell=s cell) := by
  have h := ordered_correct l r s prep valid (List.range l.count)
    (by intro p hp;simpa [prep.counts] using hp)
    (by intro p hp;simpa [prep.counts] using hp)
  refine ⟨⟨h.1,?_⟩,h.2⟩
  intro x _ hx
  apply h.2
  rintro ⟨y,hy,he⟩
  have hxy : x=y := by unfold addr at he;omega
  exact hx (hxy ▸ hy)

#axiomsClean ordered_correct
#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.VarLenCopy
