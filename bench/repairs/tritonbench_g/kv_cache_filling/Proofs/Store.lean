import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Schedule

namespace VeriTile.Bench.Repairs.KvCacheFilling

theorem runStores_frame {ν : Type} (ss : List (Store ν)) (m : Address → ν) (a : Address)
    (h : ∀ s ∈ ss, s.address ≠ a) : runStores ss m a = m a := by
  induction ss generalizing m with
  | nil => rfl
  | cons s ss ih =>
    rw [runStores, ih _ (by intro t ht; exact h t (by simp [ht]))]
    exact Function.update_of_ne (Ne.symm (h s (by simp))) _ _

theorem runStores_keep {ν : Type} (ss : List (Store ν)) (m : Address → ν) (a : Address) (v : ν)
    (hm : m a = v) (h : ∀ s ∈ ss, s.address = a → s.value = v) : runStores ss m a = v := by
  induction ss generalizing m with
  | nil => exact hm
  | cons s ss ih =>
    apply ih _ ?_ (by intro t ht; exact h t (by simp [ht]))
    by_cases hs : s.address = a
    · subst a; simpa using h s (by simp) rfl
    · simpa [Function.update_of_ne (Ne.symm hs)] using hm

theorem runStores_readback {ν : Type} (ss : List (Store ν)) (m : Address → ν) (a : Address) (v : ν)
    (present : ∃ s ∈ ss, s.address = a) (consistent : ∀ s ∈ ss, s.address = a → s.value = v) :
    runStores ss m a = v := by
  induction ss generalizing m with
  | nil => simp at present
  | cons s ss ih =>
    change runStores ss (Function.update m s.address s.value) a = v
    by_cases hs : s.address = a
    · apply runStores_keep
      · subst a; simpa using consistent s (by simp) rfl
      · intro t ht; exact consistent t (by simp [ht])
    · apply ih
      · obtain ⟨t,ht,he⟩ := present
        exact ⟨t, (List.mem_cons.mp ht).resolve_left (by intro e; subst t; exact hs he), he⟩
      · intro t ht; exact consistent t (by simp [ht])

theorem mem_taskStores {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (j : Job) (s : Store (Cell α β)) :
    s ∈ taskStores r ops kp vp k v j ↔
      s.address ∈ rowAddresses r j ∧ s.value = taskValue r ops kp vp k v j s.address := by
  simp only [taskStores, List.mem_map]
  constructor
  · rintro ⟨a,ha,e⟩; subst s; exact ⟨ha,rfl⟩
  · rintro ⟨ha,hv⟩; exact ⟨s.address,ha,by cases s; simp_all⟩

theorem whole_readback {α β : Type} (r : Request) (d : Domain r) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (m : Address → Cell α β)
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j) :
    execute r ops kp vp k v m a = taskValue r ops kp vp k v j a := by
  apply runStores_readback
  · refine ⟨⟨a,taskValue r ops kp vp k v j a⟩, ?_, rfl⟩
    apply List.mem_flatMap.mpr
    exact ⟨j,hj,(mem_taskStores ..).mpr ⟨ha,rfl⟩⟩
  · intro s hs hsa
    obtain ⟨j',hj',hs'⟩ := List.mem_flatMap.mp hs
    obtain ⟨ha',hv⟩ := (mem_taskStores ..).mp hs'
    have hrow := (mem_rowAddresses r j a).mp ha
    have hrow' := (mem_rowAddresses r j' s.address).mp ha'
    have hsame : j' = j := destinationRow_injective r d j' j hj' hj (by rw [← hrow.1, ← hrow'.1, hsa])
    simpa [hsame, hsa] using hv

theorem whole_frame {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (m : Address → Cell α β)
    (a : Address) (h : ∀ j ∈ jobs r, a ∉ rowAddresses r j) :
    execute r ops kp vp k v m a = m a := by
  apply runStores_frame
  intro s hs hsa
  obtain ⟨j,hj,hs'⟩ := List.mem_flatMap.mp hs
  have ha := ((mem_taskStores ..).mp hs').1
  exact h j hj (hsa ▸ ha)

theorem store_bound {α β : Type} (r : Request) (d : Domain r) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α)
    (s : Store (Cell α β)) (hs : s ∈ stores r ops kp vp k v) :
    s.address.row < r.pages * r.block * r.heads ∧ s.address.column < r.width s.address.role := by
  obtain ⟨j,hj,hs'⟩ := List.mem_flatMap.mp hs
  exact row_address_bound r d j hj s.address ((mem_taskStores ..).mp hs').1

end VeriTile.Bench.Repairs.KvCacheFilling
