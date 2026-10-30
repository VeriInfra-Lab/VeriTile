import bench.tritonbench_g.streamk_matmul.Proofs.Reassembly
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Memory
set_option maxHeartbeats 16000000

@[ext] theorem state_ext (s t : State α A B C S E)
    (ha : s.inputA=t.inputA) (hb : s.inputB=t.inputB) (hc : s.output=t.output)
    (hs : s.scratch=t.scratch) (he : s.exterior=t.exterior) : s=t := by
  cases s;cases t
  simp_all

/-- Any finite event permutation has the same result when its source events
commute. The premise is discharged below from source ownership and frames. -/
theorem run_permutation (add : α → α → α) (xs ys : List (Event α A B C S))
    (hp : xs.Perm ys)
    (hc : ∀ e ∈ xs, ∀ f ∈ xs, ∀ s : State α A B C S E, step add e (step add f s)=step add f (step add e s))
    (s : State α A B C S E) : run add xs s=run add ys s := by
  induction hp generalizing s with
  | nil => rfl
  | cons e hp ih =>
    change run add _ (step add e s)=run add _ (step add e s)
    apply ih
    intro e' he f hf st
    exact hc e' (List.mem_cons_of_mem _ he) f (List.mem_cons_of_mem _ hf) st
  | swap e f xs =>
    change run add xs (step add e (step add f s))=run add xs (step add f (step add e s))
    rw [hc e (by simp) f (by simp)]
  | trans h1 h2 ih1 ih2 =>
    exact (ih1 hc s).trans (ih2 (fun e he f hf st=>hc e (h1.mem_iff.mpr he) f (h1.mem_iff.mpr hf) st) s)

theorem fold_filter (xs : List β) (pred : β → Bool) (f : α → β → α) (z : α) :
    xs.foldl (fun z x=>if pred x then f z x else z) z=(xs.filter pred).foldl f z := by
  induction xs generalizing z with
  | nil => rfl
  | cons x xs ih =>
    by_cases h:pred x = Bool.true <;> simp [h,ih]
end Memory
namespace Schedule
set_option maxHeartbeats 16000000
noncomputable section

/-- Integer event indices only. Completeness forbids dropped or duplicated
source events, including any store required to initialize a returned cell. -/
structure Order (n : Nat) where
  positions : List (Fin n)
  complete : positions.Perm (List.finRange n)

def reorder (xs : List β) (order : Order xs.length) := order.positions.map xs.get

theorem reorder_perm (xs : List β) (order : Order xs.length) : (reorder xs order).Perm xs := by
  have hp:=order.complete.map xs.get
  simpa only [reorder,List.map_get_finRange] using hp

/-- Numerical-value-free reducer schedule. Each output's local instruction
order is preserved while events at different output addresses may interleave.
Geometry proves different reducer tile CTAs own disjoint output addresses. -/
structure AtomicOrder (pairs : List (Fin C × Fin S)) extends Order pairs.length where
  perOutput : ∀ i : Fin C,
    (reorder pairs toOrder).filter (fun pair=>pair.1==i)=pairs.filter (fun pair=>pair.1==i)

theorem atomic_reordering (add : α → α → α) (pairs : List (Fin C × Fin S)) (order : AtomicOrder pairs)
    (s : Memory.State α A B C S E) :
    Memory.run add ((reorder pairs order.toOrder).map fun pair=>.atomic pair.1 pair.2) s=
      Memory.run add (pairs.map fun pair=>.atomic pair.1 pair.2) s := by
  apply Memory.state_ext
  · exact (Memory.run_inputs _ _ _).1.trans (Memory.run_inputs _ _ _).1.symm
  · exact (Memory.run_inputs _ _ _).2.1.trans (Memory.run_inputs _ _ _).2.1.symm
  · funext i
    rw [Memory.atomic_projection,Memory.atomic_projection]
    have hf:(fun z pair=>if pair.1=i then add z (s.scratch pair.2) else z)=
        (fun z (pair : Fin C × Fin S) =>if pair.1==i then add z (s.scratch pair.2) else z) := by
      funext z pair
      simp
    rw [hf,Memory.fold_filter,Memory.fold_filter,order.perOutput i]
  · rw [Memory.atomic_scratch_frame,Memory.atomic_scratch_frame]
  · exact (Memory.run_inputs _ _ _).2.2.trans (Memory.run_inputs _ _ _).2.2.symm

/-- Adjacent atomic operations to distinct output addresses can be swapped
without changing any per-output integer trace. This is the cross-tile case. -/
theorem distinct_atomic_swap (before suffix : List (Fin C × Fin S)) (a b : Fin C × Fin S)
    (h : a.1≠b.1) (i : Fin C) :
    (before++a::b::suffix).filter (fun pair=>pair.1==i)=
      (before++b::a::suffix).filter (fun pair=>pair.1==i) := by
  by_cases ha:a.1=i <;> by_cases hb:b.1=i
  · exact False.elim (h (ha.trans hb.symm))
  · simp [ha,hb]
  · simp [ha,hb]
  · simp [ha,hb]
end
end Schedule
namespace Source
open Protocol Plan Schedule
set_option maxHeartbeats 16000000
noncomputable section

theorem first_only (p : Parameters) (ops : Instructions α) (e : Event α p) (he : e ∈ firstWave p ops) :
    Memory.firstOnly e := by
  unfold firstWave at he
  split at he
  next hs =>
    obtain ⟨pr,hpr,he⟩:=List.mem_flatMap.mp he
    obtain ⟨c,hc,he⟩:=List.mem_flatMap.mp he
    unfold macLoop at he
    dsimp only at he
    split at he
    next ht =>
      unfold storeTile at he
      obtain ⟨l,hl,he⟩:=List.mem_flatMap.mp he
      split at he
      next ha => simp only [List.mem_singleton] at he; subst e; trivial
      next ha => simp at he
    next ht =>
      obtain ⟨l,hl,rfl⟩:=List.mem_map.mp he
      trivial
  next hs => simp at he

/-- Source store collisions force identical events, with identical local
expressions. This fact is stronger than numerical equality. -/
theorem first_collision (p : Parameters) (ops : Instructions α) (e f : Event α p)
    (he : e ∈ firstWave p ops) (hf : f ∈ firstWave p ops) :
    (∃ i, Memory.WritesOutput i e ∧ Memory.WritesOutput i f) ∨
      (∃ i, Memory.WritesScratch i e ∧ Memory.WritesScratch i f) → e=f := by
  intro hcollision
  unfold firstWave at he
  split at he
  next hs =>
    obtain ⟨pr,hpr,he⟩:=List.mem_flatMap.mp he
    obtain ⟨c,hc,he⟩:=List.mem_flatMap.mp he
    unfold macLoop at he
    dsimp only at he
    split at he
    next ht =>
      unfold storeTile at he
      obtain ⟨lane,hl,he⟩:=List.mem_flatMap.mp he
      split at he
      next ha =>
        simp only [List.mem_singleton] at he
        subst e
        rcases hcollision with ⟨i,hi,hf'⟩|⟨i,hi,hf'⟩
        · change output p (c.tile (select p.h).iterations) lane ha=i at hi
          subst i
          exact (first_output_unique p ops hs pr c hc ht lane ha f hf hf').symm
        · simp [Memory.WritesScratch] at hi
      next ha => simp at he
    next ht =>
      obtain ⟨lane,hl,he⟩:=List.mem_map.mp he
      subst e
      rcases hcollision with ⟨i,hi,hf'⟩|⟨i,hi,hf'⟩
      · simp [Memory.WritesOutput] at hi
      · change scratch p pr lane=i at hi
        subst i
        exact (first_scratch_unique p ops hs pr c hc ht lane f hf hf').symm
  next hs => simp at he

theorem first_commute (p : Parameters) (ops : Instructions α) (e f : Event α p)
    (he : e ∈ firstWave p ops) (hf : f ∈ firstWave p ops) (s : State α p exterior) :
    Memory.step ops.atomicAdd e (Memory.step ops.atomicAdd f s)=
      Memory.step ops.atomicAdd f (Memory.step ops.atomicAdd e s) := by
  by_cases h:e=f
  · rw [h]
  · apply Memory.independent_first_steps _ e f s (first_only p ops e he) (first_only p ops f hf)
    constructor
    · intro i hi
      exact h (first_collision p ops e f he hf (Or.inl ⟨i,hi⟩))
    · intro i hi
      exact h (first_collision p ops e f he hf (Or.inr ⟨i,hi⟩))

theorem first_reordering (p : Parameters) (ops : Instructions α)
    (order : Order (firstWave p ops).length) (initial : State α p exterior) :
    Memory.run ops.atomicAdd (reorder (firstWave p ops) order) initial=
      Memory.run ops.atomicAdd (firstWave p ops) initial := by
  apply Memory.run_permutation _ _ _ (reorder_perm _ order)
  intro e he f hf s
  have hp:=reorder_perm (firstWave p ops) order
  exact first_commute p ops e f (hp.mem_iff.mp he) (hp.mem_iff.mp hf) s


theorem full_only (p : Parameters) (ops : Instructions α) (e : Event α p) (he : e ∈ fullWave p ops) :
    Memory.firstOnly e := by
  unfold fullWave at he
  obtain ⟨pid,hpid,he⟩:=List.mem_flatMap.mp he
  unfold storeTile at he
  obtain ⟨lane,hl,he⟩:=List.mem_flatMap.mp he
  split at he
  next ha => simp only [List.mem_singleton] at he; subst e; trivial
  next ha => simp at he

theorem full_commute (p : Parameters) (ops : Instructions α) (e f : Event α p)
    (he : e ∈ fullWave p ops) (hf : f ∈ fullWave p ops) (s : State α p exterior) :
    Memory.step ops.atomicAdd e (Memory.step ops.atomicAdd f s)=
      Memory.step ops.atomicAdd f (Memory.step ops.atomicAdd e s) := by
  by_cases h:e=f
  · rw [h]
  · have hone:=full_only p ops e he
    have htwo:=full_only p ops f hf
    apply Memory.independent_first_steps _ e f s hone htwo
    unfold fullWave at he
    obtain ⟨pid,hpid,he⟩:=List.mem_flatMap.mp he
    unfold storeTile at he
    obtain ⟨lane,hl,he⟩:=List.mem_flatMap.mp he
    split at he
    next ha =>
      simp only [List.mem_singleton] at he
      subst e
      constructor
      · intro i hi
        have hii:output p (pid+(select p.h).streamTiles) lane ha=i:=hi.1
        subst i
        have hpid':pid < (select p.h).blockingTiles:=List.mem_range.mp hpid
        have hsum:=selected_partition p.h
        have bound:pid+(select p.h).streamTiles < tiles p.h:=by omega
        exact h (full_output_unique p ops _ bound lane ha f hf hi.2).symm
      · intro i hi
        exact hi.1
    next ha => simp at he

theorem full_reordering (p : Parameters) (ops : Instructions α)
    (order : Order (fullWave p ops).length) (initial : State α p exterior) :
    Memory.run ops.atomicAdd (reorder (fullWave p ops) order) initial=
      Memory.run ops.atomicAdd (fullWave p ops) initial := by
  apply Memory.run_permutation _ _ _ (reorder_perm _ order)
  intro e he f hf s
  have hp:=reorder_perm (fullWave p ops) order
  exact full_commute p ops e f (hp.mem_iff.mp he) (hp.mem_iff.mp hf) s

#axiomsClean full_reordering

#axiomsClean first_commute
#axiomsClean first_reordering
#axiomsClean Schedule.atomic_reordering
#specNonCircular Schedule.AtomicOrder avoiding [call,Memory.run,Memory.step]
end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001
