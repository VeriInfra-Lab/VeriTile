import bench.tritonbench_g.matmul_dequant_int4.Sources.Helpers
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.HelperReal
open Helpers
set_option maxHeartbeats 5000000
noncomputable section

/-- Mathematical nearest-even rounding followed by unsigned nibble clamping. -/
def nearestEven (x : ℝ) : Int :=
  let low := Int.floor x
  if x-(low:ℝ)<(1/2:ℝ) then low
  else if x-(low:ℝ)>(1/2:ℝ) then low+1
  else if low%2=0 then low else low+1

def clampNibble (x : ℝ) : BitVec 4 := BitVec.ofInt 4 (max 0 (min 15 (nearestEven x)))
def arithmetic : Arithmetic ℝ :=
  ⟨0,(fun x => x / 15),(fun x => x),(-·),(·+·),(·-·),(·*·),(·/·),max,min,(max 0),(min 0),clampNibble⟩

/-- Independent range objective: extrema of the logical group extended with zero. -/
structure GroupRange {g : Nat} (input : Fin g → ℝ) (hi lo : ℝ) : Prop where
  highZero : 0≤hi
  lowZero : lo≤0
  upper : ∀ i,input i≤hi
  lower : ∀ i,lo≤ input i
  highTight : hi=0 ∨ ∃ i,hi=input i
  lowTight : lo=0 ∨ ∃ i,lo=input i

theorem tree_max_bound (tree : ReductionTree g) (input : Fin g → ℝ) (i : Fin g) (hi : i∈tree.leaves) :
    input i≤tree.eval max input := by
  induction tree with
  | leaf j =>
    have he : i=j := by simpa [ReductionTree.leaves] using hi
    subst i
    exact le_rfl
  | node l r ihl ihr =>
    rcases List.mem_append.mp hi with h|h
    · exact (ihl h).trans (le_max_left _ _)
    · exact (ihr h).trans (le_max_right _ _)

theorem tree_min_bound (tree : ReductionTree g) (input : Fin g → ℝ) (i : Fin g) (hi : i∈tree.leaves) :
    tree.eval min input≤ input i := by
  induction tree with
  | leaf j =>
    have he : i=j := by simpa [ReductionTree.leaves] using hi
    subst i
    exact le_rfl
  | node l r ihl ihr =>
    rcases List.mem_append.mp hi with h|h
    · exact (min_le_left _ _).trans (ihl h)
    · exact (min_le_right _ _).trans (ihr h)

theorem tree_max_attained (tree : ReductionTree g) (input : Fin g → ℝ) :
    ∃ i,tree.eval max input=input i := by
  induction tree with
  | leaf i => exact ⟨i,rfl⟩
  | node l r ihl ihr =>
    rcases le_total (l.eval max input) (r.eval max input) with h|h
    · simpa [ReductionTree.eval,max_eq_right h] using ihr
    · simpa [ReductionTree.eval,max_eq_left h] using ihl

theorem tree_min_attained (tree : ReductionTree g) (input : Fin g → ℝ) :
    ∃ i,tree.eval min input=input i := by
  induction tree with
  | leaf i => exact ⟨i,rfl⟩
  | node l r ihl ihr =>
    rcases le_total (l.eval min input) (r.eval min input) with h|h
    · simpa [ReductionTree.eval,min_eq_left h] using ihl
    · simpa [ReductionTree.eval,min_eq_right h] using ihr

theorem stats_range (s : Shape) (order : ReductionOrder s.group) (initial : State ℝ) (g n : Nat) :
    GroupRange (fun i : Fin s.group => input s initial g n i)
      (groupStats s arithmetic order initial g n).1 (groupStats s arithmetic order initial g n).2 := by
  let values := fun i : Fin s.group => input s initial g n i
  change GroupRange values (max 0 (order.tree.eval max values)) (min 0 (order.tree.eval min values))
  have mem (i : Fin s.group) : i∈order.tree.leaves := order.complete.mem_iff.mpr (by simp)
  refine ⟨le_max_left _ _,min_le_left _ _,?_,?_,?_,?_⟩
  · intro i;exact (tree_max_bound order.tree values i (mem i)).trans (le_max_right _ _)
  · intro i;exact (min_le_right _ _).trans (tree_min_bound order.tree values i (mem i))
  · by_cases h : order.tree.eval max values≤0
    · exact Or.inl (max_eq_left h)
    · obtain ⟨i,hi⟩ := tree_max_attained order.tree values
      exact Or.inr ⟨i,(max_eq_right (by linarith)).trans hi⟩
  · by_cases h : 0≤order.tree.eval min values
    · exact Or.inl (min_eq_left h)
    · obtain ⟨i,hi⟩ := tree_min_attained order.tree values
      exact Or.inr ⟨i,(min_eq_right (by linarith)).trans hi⟩

theorem range_unique {g : Nat} (input : Fin g → ℝ) (hi lo hi' lo' : ℝ)
    (h : GroupRange input hi lo) (h' : GroupRange input hi' lo') : hi=hi' ∧ lo=lo' := by
  have hh : hi≤hi' := by rcases h.highTight with rfl|⟨i,rfl⟩;exact h'.highZero;exact h'.upper i
  have hh' : hi'≤hi := by rcases h'.highTight with rfl|⟨i,rfl⟩;exact h.highZero;exact h.upper i
  have hl : lo≤lo' := by rcases h'.lowTight with rfl|⟨i,rfl⟩;exact h.lowZero;exact h.lower i
  have hl' : lo'≤lo := by rcases h.lowTight with rfl|⟨i,rfl⟩;exact h'.lowZero;exact h'.lower i
  exact ⟨le_antisymm hh hh',le_antisymm hl hl'⟩

theorem scale_independent (s : Shape) (order : ReductionOrder s.group) (initial : State ℝ)
    (g n : Nat) (hi lo : ℝ) (range : GroupRange (fun i : Fin s.group => input s initial g n i) hi lo) :
    scale s arithmetic order initial g n=(hi-lo)/15 := by
  have he := range_unique _ _ _ hi lo (stats_range s order initial g n) range
  change ((groupStats s arithmetic order initial g n).1-(groupStats s arithmetic order initial g n).2)/15 = _
  rw [he.1,he.2]

def mathZero (hi lo : ℝ) := clampNibble (-lo/((hi-lo)/15))
def mathWeight (value hi lo : ℝ) := clampNibble (value/((hi-lo)/15)+((mathZero hi lo).toNat:ℝ))

theorem zero_independent (s : Shape) (order : ReductionOrder s.group) (initial : State ℝ)
    (g n : Nat) (hi lo : ℝ) (range : GroupRange (fun i : Fin s.group => input s initial g n i) hi lo) :
    zeroNibble s arithmetic order initial g n=mathZero hi lo := by
  have he := range_unique _ _ _ hi lo (stats_range s order initial g n) range
  change clampNibble (-(groupStats s arithmetic order initial g n).2/(scale s arithmetic order initial g n)) = _
  rw [he.2,scale_independent s order initial g n hi lo range]
  rfl

theorem weight_independent (s : Shape) (order : ReductionOrder s.group) (initial : State ℝ)
    (k n : Nat) (hi lo : ℝ) (range : GroupRange (fun i : Fin s.group => input s initial (k/s.group) n i) hi lo) :
    weightNibble s arithmetic order initial k n=mathWeight (initial.values .weight (k*s.n+n)) hi lo := by
  change clampNibble (initial.values .weight (k*s.n+n)/(scale s arithmetic order initial (k/s.group) n)+
    (((zeroNibble s arithmetic order initial (k/s.group) n).toNat:Int):ℝ)) = _
  rw [scale_independent s order initial (k/s.group) n hi lo range,
    zero_independent s order initial (k/s.group) n hi lo range]
  simp [mathWeight]


#axiomsClean zero_independent
#axiomsClean weight_independent
#axiomsClean stats_range
#axiomsClean range_unique
#axiomsClean scale_independent
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.HelperReal
