import bench.tritonbench_g.bmm_optimized.Sources.Bounds

namespace VeriTile.Bench.Repairs.BmmOptimized
variable {α : Type}

/-- A repeated write is harmless only after proving its value agrees. This
lemma does not assume the grid or lane list is duplicate-free. -/
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

theorem programWrites_member (s : Shape) (c : Config) (p : Scalar α)
    (pid : Program) (m : Memory α) (w : Nat×α) :
    w∈programWrites s c p pid m ↔
      ∃ lane∈lanes c,∃ x : Cell, cellOf s c pid lane=x ∧ s.Active x ∧
        w=(s.o x.batch x.row x.column,sourceValue s p m x) := by
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

def allWrites (s : Shape) (c : Config) (p : Scalar α) (m : Memory α) :=
  (programs s c).flatMap (fun pid => programWrites s c p pid m)

theorem allWrites_member (s : Shape) (c : Config) (p : Scalar α)
    (m : Memory α) (w : Nat×α) (hw : w∈allWrites s c p m) :
    ∃ x : Cell,s.Active x ∧ w=(s.o x.batch x.row x.column,sourceValue s p m x) := by
  obtain ⟨pid,_,hwrite⟩ := List.mem_flatMap.mp hw
  obtain ⟨_,_,x,_,hx,eq⟩ := (programWrites_member s c p pid m w).mp hwrite
  exact ⟨x,hx,eq⟩

theorem allWrites_covers (s : Shape) (c : Config) (p : Scalar α)
    (m : Memory α) (x : Cell) (hx : s.Active x) :
    (s.o x.batch x.row x.column,sourceValue s p m x)∈allWrites s c p m := by
  obtain ⟨pid,hpid,lane,hlane,eq⟩ := source_grid_covers s c x hx
  apply List.mem_flatMap.mpr
  refine ⟨pid,hpid,?_⟩
  exact (programWrites_member s c p pid m _).mpr ⟨lane,hlane,x,eq,hx,rfl⟩

/-- Whole actual source grid readback, after current-memory execution and
source-specific swizzle coverage. No final-output premise occurs. -/
theorem source_read (s : Shape) (c : Config) (p : Scalar α)
    (m : Memory α) (x : Cell) (hx : s.Active x) :
    source s c p m .output (s.o x.batch x.row x.column)=sourceValue s p m x := by
  rw [source,sourcePrograms_commit]
  apply commit_read
  · exact ⟨_,allWrites_covers s c p m x hx,rfl⟩
  · intro w hw address
    obtain ⟨y,hy,eq⟩ := allWrites_member s c p m w hw
    subst w
    have same := address_injective s y x hy hx address
    simpa [same]

/-- Every other allocation and every non-output-range address is preserved. -/
theorem source_frame (s : Shape) (c : Config) (p : Scalar α)
    (m : Memory α) (r : Region) (address : Nat)
    (outside : r≠.output ∨ s.size≤address) :
    source s c p m r address=m r address := by
  rw [source,sourcePrograms_commit]
  apply commit_frame
  intro w hw
  rcases outside with region|bound
  · exact Or.inl region
  · obtain ⟨x,hx,eq⟩ := allWrites_member s c p m w hw
    have upper := address_bound s x hx
    subst w
    exact Or.inr (by simp only;omega)

theorem source_grid_ordered (s : Shape) (c : Config) (p : Scalar α)
    (m : Memory α) (x : Cell) (hx : s.Active x) :
    source s c p m .output (s.o x.batch x.row x.column)=ordered s p m x := by
  rw [source_read s c p m x hx,source_ordered s p m x hx]

theorem source_grid_real (s : Shape) (c : Config)
    (m : Memory ℝ) (x : Cell) (hx : s.Active x) :
    source s c realScalar m .output (s.o x.batch x.row x.column)=mathematical s m x := by
  rw [source_grid_ordered s c realScalar m x hx,ordered_real]

#axiomsClean commit_read
#axiomsClean allWrites_covers
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean source_grid_ordered
#axiomsClean source_grid_real
end VeriTile.Bench.Repairs.BmmOptimized
