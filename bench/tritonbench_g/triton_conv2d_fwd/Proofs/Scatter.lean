import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Memory
namespace VeriTile.Bench.Repairs.Conv2d
variable {α β : Type}
theorem commit_read (writes : List (Nat×α)) (m : Memory α) (address : Nat) (value : α)
    (present : ∃ w∈writes,w.1=address)
    (agree : ∀ w∈writes,w.1=address → w.2=value) :
    commit writes m .output address=value := by
  induction writes generalizing m with
  | nil => simpa using present
  | cons w ws ih =>
    change commit ws (store m w.1 w.2) .output address=value
    by_cases tail : ∃ v∈ws,v.1=address
    · exact ih _ tail (fun v hv => agree v (by simp [hv]))
    · have here : w.1=address := by
        rcases present with ⟨v,hv,eq⟩
        rcases List.mem_cons.mp hv with same|rest
        · simpa [same] using eq
        · exact False.elim (tail ⟨v,rest,eq⟩)
      rw [commit_frame _ _ _ _ (by
        intro v hv
        exact Or.inr (fun h => tail ⟨v,hv,h.symm⟩))]
      simp only [store,here,and_self,↓reduceIte]
      exact agree w (by simp) here

theorem programWrites_member (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags)
    (pid : Program) (m : Memory α) (w : Nat×α) :
    w∈programWrites s p flags pid m ↔
      ∃ lane∈lanes,∃ x : Cell, cellOf pid lane=x ∧ s.Active x ∧
        w=(outputPointer s x,sourceValue s p flags m x) := by
  unfold programWrites
  rw [List.mem_filterMap]
  constructor
  · rintro ⟨x,hx,hw⟩
    obtain ⟨lane,hl,eq⟩ := List.mem_map.mp hx
    by_cases active : s.Active x
    · simp only [active,↓reduceIte,Option.some.injEq] at hw
      exact ⟨lane,hl,x,eq,active,hw.symm⟩
    · simp [active] at hw
  · rintro ⟨lane,hl,x,eq,active,hw⟩
    refine ⟨x,List.mem_map.mpr ⟨lane,hl,eq⟩,?_⟩
    simp [active,hw]

def allWrites (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags) (m : Memory α) :=
  (programs s).flatMap (fun pid => programWrites s p flags pid m)

theorem allWrites_member (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags)
    (m : Memory α) (w : Nat×α) (hw : w∈allWrites s v p flags m) :
    ∃ x : Cell,s.Active x ∧ w=(outputPointer s x,sourceValue s p flags m x) := by
  obtain ⟨pid,_,hwrite⟩ := List.mem_flatMap.mp hw
  obtain ⟨_,_,x,_,hx,eq⟩ := (programWrites_member s v p flags pid m w).mp hwrite
  exact ⟨x,hx,eq⟩

theorem allWrites_covers (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags)
    (m : Memory α) (x : Cell) (hx : s.Active x) :
    (outputPointer s x,sourceValue s p flags m x)∈allWrites s v p flags m := by
  obtain ⟨pid,hpid,lane,hlane,eq⟩ := source_grid_covers s v x hx
  apply List.mem_flatMap.mpr
  refine ⟨pid,hpid,?_⟩
  exact (programWrites_member s v p flags pid m _).mpr ⟨lane,hlane,x,eq,hx,rfl⟩

/-- Whole actual source grid readback, after current-memory execution and
source-specific swizzle coverage. No final-output premise occurs. -/
theorem source_read (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags)
    (m : Memory α) (x : Cell) (hx : s.Active x) :
    source s p flags m .output (outputPointer s x)=sourceValue s p flags m x := by
  rw [source,sourcePrograms_commit]
  apply commit_read
  · exact ⟨_,allWrites_covers s v p flags m x hx,rfl⟩
  · intro w hw address
    obtain ⟨y,hy,eq⟩ := allWrites_member s v p flags m w hw
    subst w
    have same := address_injective s v y x hy hx address
    simpa [same]

/-- Every other allocation and every non-output-range address is preserved. -/
theorem source_frame (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags)
    (m : Memory α) (r : Region) (address : Nat)
    (outside : r≠.output ∨ s.size≤address) :
    source s p flags m r address=m r address := by
  rw [source,sourcePrograms_commit]
  apply commit_frame
  intro w hw
  rcases outside with region|bound
  · exact Or.inl region
  · obtain ⟨x,hx,eq⟩ := allWrites_member s v p flags m w hw
    have upper := address_bound s v x hx
    subst w
    exact Or.inr (by simp only;omega)


#axiomsClean source_read
#axiomsClean source_frame
end VeriTile.Bench.Repairs.Conv2d
