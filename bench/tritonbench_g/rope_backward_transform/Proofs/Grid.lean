import bench.tritonbench_g.rope_backward_transform.Proofs.Row
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def scalarResult {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m : Source.Memory α) (key : Bool) (row h c : Nat) : α :=
  if c<2*r.half then
    let j := if c<r.half then c else c-r.half
    scalarPair op (decide (r.half≤c))
      ((m (region key) (r.offset key row h j)).read .fp32)
      ((m (region key) (r.offset key row h (j+r.half))).read .fp32)
      ((m "C" (row%r.sequence*r.cosRowStride+j*r.cosColStride)).read .fp32)
      ((m "S" (row%r.sequence*r.sinRowStride+j*r.sinColStride)).read .fp32)
  else (m (region key) (r.offset key row h c)).read .fp32

theorem region_injective : Function.Injective region := by
  intro a b h
  cases a <;> cases b <;> simp [region] at h ⊢

theorem outside_coordinate (r : Request) (row a h c : Nat) (key : Bool)
    (hh : h<r.heads key) (hc : c<r.dimension) (other : row≠a ∨ 2*r.half≤c) :
    OutsideRow r row (region key) (r.offset key a h c) := by
  intro k side
  by_cases hk : key=k
  · subst k
    right
    intro i hi eq
    have ha := (active_iff _ _ i).mp hi
    unfold address at eq
    rw [address_coordinate] at eq
    have same := offset_injective r key hh hc ha.1 (half_column r side _ ha.2) eq
    rcases other with hn | ht
    · exact hn same.1.symm
    · have hj := ha.2
      cases side <;> simp only [Bool.false_eq_true,↓reduceIte] at same <;> omega
  · exact Or.inl (fun eq => hk (region_injective eq))

theorem row_other {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (row a h c : Nat) (key : Bool)
    (hh : h<r.heads key) (hc : c<r.dimension) (other : row≠a ∨ 2*r.half≤c) :
    rowMemory M r row m (region key) (r.offset key a h c)=m (region key) (r.offset key a h c) :=
  row_frame M r m row _ _ (outside_coordinate r row a h c key hh hc other)

theorem row_coefficient {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (row o : Nat) (name : RegionName) (hn : name="C" ∨ name="S") :
    rowMemory M r row m name o=m name o := by
  apply row_frame
  intro key side
  left
  rcases hn with rfl | rfl <;> cases key <;> decide

theorem row_coordinate {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m : Source.Memory α) (row h c : Nat) (key : Bool)
    (hh : h<r.heads key) (hc : c<r.dimension) :
    (rowMemory op.backend r row m (region key) (r.offset key row h c)).read .fp32=
      scalarResult op r m key row h c := by
  by_cases ht : c<2*r.half
  · let j := if c<r.half then c else c-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    let i : Lane r key := (⟨h,lt_of_lt_of_le hh (pad_covers _)⟩,
      ⟨j,lt_of_lt_of_le hj (half_covers r)⟩,PUnit.unit)
    have hi : Source.active (r.heads key) r.half i := (active_iff _ _ _).mpr ⟨hh,hj⟩
    let side := decide (r.half≤c)
    have he : address r row key side i=r.offset key row h c := by
      unfold address
      rw [address_coordinate]
      congr 1
      dsimp [i,j,side]
      split <;> simp_all <;> omega
    rw [←he,row_readback op r m row key side i hi]
    change value op r m row key side i=_
    rw [value_eq op r m row key side i hi]
    simp only [scalarResult,ht,↓reduceIte]
    simp [address,address_coordinate,i,j,side]
  · rw [row_other op.backend r m row row h c key hh hc (Or.inr (by omega))]
    simp [scalarResult,ht]

/-- The list records completion order from last to first. Each step executes the
entire source program against the memory produced by all preceding programs. -/
def grid {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) : List Nat→Source.Memory α
  | [] => m
  | a::xs => rowMemory M r a (grid M r m xs)

inductive GridExec {α : Type} [Inhabited α] (M : Algebra α) (r : Request) :
    List Nat→Source.Memory α→Source.Memory α→Prop
  | nil (m) : GridExec M r [] m m
  | cons {xs m current after} (row : Nat) : GridExec M r xs m current →
      (∃ t,FP.Structural.exec M (Source.kernel .fp32 .fp32 (parameters r)) (state current row)=some t ∧ t.mem=after) →
      GridExec M r (row::xs) m after

theorem grid_execution {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (xs : List Nat) : GridExec M r xs m (grid M r m xs) := by
  induction xs with
  | nil => exact .nil m
  | cons a xs ih => exact .cons a ih (row_execution M r a _)

theorem grid_frame {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (xs : List Nat) (name : RegionName) (o : Nat)
    (h : ∀ row∈xs,OutsideRow r row name o) : grid M r m xs name o=m name o := by
  induction xs with
  | nil => rfl
  | cons a xs ih =>
    rw [grid,row_frame M r _ a name o (h a (by simp))]
    exact ih (fun row hr => h row (by simp [hr]))

theorem grid_other {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (xs : List Nat) (a h c : Nat) (key : Bool)
    (hh : h<r.heads key) (hc : c<r.dimension) (other : a∉xs ∨ 2*r.half≤c) :
    grid M r m xs (region key) (r.offset key a h c)=m (region key) (r.offset key a h c) := by
  apply grid_frame
  intro row hr
  apply outside_coordinate r row a h c key hh hc
  rcases other with hn | ht
  · left; intro eq; exact hn (eq ▸ hr)
  · exact Or.inr ht

theorem grid_coefficient {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (xs : List Nat) (o : Nat) (name : RegionName) (hn : name="C" ∨ name="S") :
    grid M r m xs name o=m name o := by
  induction xs with
  | nil => rfl
  | cons a xs ih => rw [grid,row_coefficient M r _ a o name hn,ih]

theorem result_current {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m : Source.Memory α) (xs : List Nat) (a h c : Nat) (key : Bool)
    (hh : h<r.heads key) (hc : c<r.dimension) (ha : a∉xs) :
    scalarResult op r (grid op.backend r m xs) key a h c=scalarResult op r m key a h c := by
  unfold scalarResult
  split
  · rename_i ht
    let j := if c<r.half then c else c-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    have h0 := half_column r Bool.false j hj
    have h1 := half_column r Bool.true j hj
    simp only [Bool.false_eq_true,↓reduceIte,Nat.add_zero] at h0 h1
    dsimp only
    rw [grid_other op.backend r m xs a h j key hh h0 (Or.inl ha),
      grid_other op.backend r m xs a h (j+r.half) key hh h1 (Or.inl ha),
      grid_coefficient op.backend r m xs _ "C" (Or.inl rfl),
      grid_coefficient op.backend r m xs _ "S" (Or.inr rfl)]
  · rw [grid_other op.backend r m xs a h c key hh hc (Or.inl ha)]

theorem grid_readback {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m : Source.Memory α) (xs : List Nat) (hn : xs.Nodup)
    (a h c : Nat) (key : Bool) (ha : a∈xs) (hh : h<r.heads key) (hc : c<r.dimension) :
    (grid op.backend r m xs (region key) (r.offset key a h c)).read .fp32=
      scalarResult op r m key a h c := by
  induction xs with
  | nil => simp at ha
  | cons row xs ih =>
    obtain ⟨hnot,hnd⟩ := List.nodup_cons.mp hn
    by_cases eq : row=a
    · subst row
      rw [grid,row_coordinate op r _ a h c key hh hc,result_current op r m xs a h c key hh hc hnot]
    · rw [grid,row_other op.backend r _ row a h c key hh hc (Or.inl eq)]
      exact ih hnd ((List.mem_cons.mp ha).resolve_left (Ne.symm eq))

#axiomsClean row_coordinate
#axiomsClean grid_execution
#axiomsClean grid_readback
#axiomsClean grid_frame
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
