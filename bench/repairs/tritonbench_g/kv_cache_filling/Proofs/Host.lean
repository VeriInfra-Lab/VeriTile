import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Schedule

namespace VeriTile.Bench.Repairs.KvCacheFilling

def tokens (r : Request) : List (Nat × Nat) :=
  (List.range r.batches).flatMap fun b => (List.range (r.count b)).map fun t => (b,t)

theorem mem_tokens (r : Request) (bt : Nat × Nat) :
    bt ∈ tokens r ↔ bt.1 < r.batches ∧ bt.2 < r.count bt.1 := by
  rcases bt with ⟨b,t⟩
  simp [tokens]

theorem tokens_nodup (r : Request) : (tokens r).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro b hb
    apply List.Nodup.map
    · intro a b he; exact (Prod.mk.inj he).2
    · exact List.nodup_range
  · apply List.Pairwise.imp_of_mem ?_ List.nodup_range
    intro b b' hb hb' hne
    apply List.disjoint_left.mpr
    intro bt hbt hbt'
    obtain ⟨t,_,he⟩ := List.mem_map.mp hbt
    obtain ⟨t',_,he'⟩ := List.mem_map.mp hbt'
    exact hne (congrArg Prod.fst (he.trans he'.symm))

def destinations (r : Request) : List Nat := (tokens r).map fun bt => destinationToken r bt.1 bt.2

theorem destinations_nodup (r : Request) (d : Domain r) : (destinations r).Nodup := by
  apply List.Nodup.map_on ?_ (tokens_nodup r)
  intro a ha b hb he
  have h := d.token_ownership a.1 a.2 b.1 b.2 ((mem_tokens r a).mp ha).1 ((mem_tokens r a).mp ha).2
    ((mem_tokens r b).mp hb).1 ((mem_tokens r b).mp hb).2 he
  exact Prod.ext h.1 h.2

/-- The actual occupied-set insertion loop, with no acceptance premise. -/
def reserve : List Nat → List Nat → Option (List Nat)
  | [],used => some used
  | x::xs,used => if x ∈ used then none else reserve xs (x::used)

theorem reserve_success (xs used : List Nat) (hn : xs.Nodup) (hd : xs.Disjoint used) :
    ∃ final, reserve xs used = some final := by
  induction xs generalizing used with
  | nil => exact ⟨used,rfl⟩
  | cons x xs ih =>
    obtain ⟨hx,hn'⟩ := List.nodup_cons.mp hn
    have hxu : x ∉ used := by intro h; exact List.disjoint_left.mp hd (by simp) h
    have hdu : xs.Disjoint (x::used) := by
      apply List.disjoint_left.mpr
      intro a ha hau
      rcases List.mem_cons.mp hau with he | he
      · subst a; exact hx ha
      · exact List.disjoint_left.mp hd (by simp [ha]) he
    obtain ⟨final,hf⟩ := ih (x::used) hn' hdu
    exact ⟨final,by simp [reserve,hxu,hf]⟩

theorem host_ownership_check (r : Request) (d : Domain r) :
    ∃ final, reserve (destinations r) [] = some final :=
  reserve_success _ _ (destinations_nodup r d) (by simp)

def sourceRows (r : Request) : List Nat := (jobs r).map (sourceRow r)
def destinationRows (r : Request) : List Nat := (jobs r).map (destinationRow r)

theorem schedule_lengths (r : Request) :
    (sourceRows r).length = (jobs r).length ∧ (destinationRows r).length = (jobs r).length := by
  simp [sourceRows,destinationRows]

theorem launch_program (r : Request) (i : Fin (jobs r).length) :
    (sourceRows r)[i.val]'(by simp [sourceRows]) = sourceRow r ((jobs r).get i) ∧
    (destinationRows r)[i.val]'(by simp [destinationRows]) = destinationRow r ((jobs r).get i) := by
  simp [sourceRows,destinationRows]

theorem every_job_launched (r : Request) (j : Job) (hj : j ∈ jobs r) :
    ∃ i : Fin (jobs r).length, (jobs r).get i = j := List.mem_iff_get.mp hj

theorem host_active_checks (r : Request) (d : Domain r) (bt : Nat × Nat) (h : bt ∈ tokens r) :
    r.count bt.1 ≤ r.maxQ ∧ r.start bt.1+r.count bt.1 ≤ r.tokens ∧ r.count bt.1 ≤ r.total bt.1 ∧
    (r.history bt.1+bt.2)/r.block < r.tableWidth ∧ r.page bt.1 ((r.history bt.1+bt.2)/r.block) < r.pages := by
  have hh := (mem_tokens r bt).mp h
  exact ⟨(d.lengths bt.1 hh.1).1,(d.lengths bt.1 hh.1).2.1,(d.lengths bt.1 hh.1).2.2,
    (d.page_bounds bt.1 bt.2 hh.1 hh.2).1,(d.page_bounds bt.1 bt.2 hh.1 hh.2).2⟩


theorem jobs_by_tokens (r : Request) :
    jobs r = (tokens r).flatMap (fun bt => (List.range r.heads).map fun h => (⟨bt.1,bt.2,h⟩ : Job)) := by
  simp [jobs,tokens,List.flatMap_assoc,List.flatMap_map]

theorem jobs_nodup (r : Request) : (jobs r).Nodup := by
  rw [jobs_by_tokens]
  apply List.nodup_flatMap.mpr
  constructor
  · intro bt hbt
    apply List.Nodup.map
    · intro a b he; exact congrArg Job.head he
    · exact List.nodup_range
  · apply List.Pairwise.imp_of_mem ?_ (tokens_nodup r)
    intro bt bt' hbt hbt' hne
    apply List.disjoint_left.mpr
    intro j hj hj'
    obtain ⟨h,_,he⟩ := List.mem_map.mp hj
    obtain ⟨h',_,he'⟩ := List.mem_map.mp hj'
    have hc := he.trans he'.symm
    exact hne (Prod.ext (congrArg Job.batch hc) (congrArg Job.token hc))

end VeriTile.Bench.Repairs.KvCacheFilling
