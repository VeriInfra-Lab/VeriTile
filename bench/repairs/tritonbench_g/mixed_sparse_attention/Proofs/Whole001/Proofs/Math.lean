import VeriTile.Triton.Math.Attention
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget
open VeriTile.Triton
noncomputable section

/-- Logical count-selected support, independent of the implementation. Entries
are occurrences; duplicates have their ordinary multiplicity. -/
def support (BN length query blocks columns : Nat)
    (blockOffset columnIndex : Nat → Nat) : List Nat :=
  ((List.range blocks).flatMap fun b =>
    ((List.range BN).map fun j => blockOffset b+j).filter fun n =>
      decide (n<length ∧ n≤query)) ++
    (List.range columns).map columnIndex

def score (D : Nat) (scale : ℝ) (q k : Nat → ℝ) : ℝ :=
  (scale*1.44269504)*∑ d∈Finset.range D,q d*k d

/-- Independent input-only sparse attention. Real division gives zero on
empty support; every nonempty support has a strictly positive denominator. -/
def expected (D BN length query blocks columns : Nat) (scale : ℝ)
    (q : Nat → ℝ) (k v : Nat → Nat → ℝ)
    (blockOffset columnIndex : Nat → Nat) (channel : Nat) : ℝ :=
  let keys:=support BN length query blocks columns blockOffset columnIndex
  (keys.map (fun n=>pow2 (score D scale q (k n))*v n channel)).sum /
    (keys.map (fun n=>pow2 (score D scale q (k n)))).sum

theorem selected_column (BN length query blocks columns : Nat)
    (blockOffset columnIndex : Nat → Nat) (j : Nat) (hj : j<columns) :
    columnIndex j∈support BN length query blocks columns blockOffset columnIndex := by
  unfold support
  apply List.mem_append_right
  exact List.mem_map.mpr ⟨j,List.mem_range.mpr hj,rfl⟩

theorem selected_block (BN length query blocks columns : Nat)
    (blockOffset columnIndex : Nat → Nat) (b j : Nat)
    (hb : b<blocks) (hj : j<BN) (hn : blockOffset b+j<length)
    (hc : blockOffset b+j≤query) :
    blockOffset b+j∈support BN length query blocks columns blockOffset columnIndex := by
  unfold support
  apply List.mem_append_left
  apply List.mem_flatMap.mpr
  refine ⟨b,List.mem_range.mpr hb,?_⟩
  exact List.mem_filter.mpr ⟨List.mem_map.mpr ⟨j,List.mem_range.mpr hj,rfl⟩,by simp [hn,hc]⟩

theorem positive_mass (keys : List Nat) (s : Nat → ℝ) (hne : keys≠[]) :
    0<(keys.map (fun n=>pow2 (s n))).sum := by
  induction keys with
  | nil=>simp at hne
  | cons n ns ih=>
    simp only [List.map_cons,List.sum_cons]
    have htail : 0≤(ns.map (fun j=>pow2 (s j))).sum := by
      exact List.sum_nonneg (by intro x hx; obtain ⟨j,_,rfl⟩:=List.mem_map.mp hx; exact le_of_lt (pow2_pos _))
    exact add_pos_of_pos_of_nonneg (pow2_pos _) htail

theorem empty_support (D BN length query : Nat) (scale : ℝ)
    (q : Nat → ℝ) (k v : Nat → Nat → ℝ) (offset index : Nat → Nat) (d : Nat) :
    expected D BN length query 0 0 scale q k v offset index d=0 := by
  simp [expected,support]

#axiomsClean selected_column
#axiomsClean selected_block
#axiomsClean positive_mass
#axiomsClean empty_support
#specNonCircular expected avoiding [VeriTile.Triton.osBlockStep]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget
