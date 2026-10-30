import bench.tritonbench_g.layer_norm_fwd.Proofs.FlatDomain
import bench.tritonbench_g.layer_norm_fwd.Proofs.Real

/-! Independent finite-memory image of the seven-output contract. Every cell
is specified, including untouched boolean/integer/pointer tags, translated
pointer values, unallocated addresses, and other physical regions. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.FlatContract
open VeriTile.Triton
noncomputable section

/-- Initial-snapshot mathematical output, translated into the finite allocation.
The last-store-first contract preserves source output alias order. -/
def expected (A : FlatAlloc) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial : BlockState) (region : RegionName) (offset : Nat) : MemCell :=
  if region = A.flat then
    match A.decode offset with
    | none => MemCell.real 0
    | some (q, z) => A.trCell
        ((Contract.expectedCell f r stride (snapshot r stride initial) M N eps p initial.pid q z).getD (initial.mem q z))
  else MemCell.real 0

def Post (A : FlatAlloc) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState) : Prop :=
  ∀ region offset, final.mem region offset = expected A f r stride M N eps p initial region offset

def Outside (A : FlatAlloc) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (initial : BlockState) (region : RegionName) (offset : Nat) : Prop :=
  region ≠ A.flat ∨ ∀ q z, Contract.written f r stride M N initial.pid q z → offset ≠ A.addr q z

theorem flatten_post (A : FlatAlloc) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState)
    (h : Contract.Post f r stride M N eps p initial final) :
    Post A f r stride M N eps p initial (A.flattenState final) := by
  intro region offset
  simp only [expected, FlatAlloc.flattenState, FlatAlloc.readFlat]
  split
  · cases A.decode offset with
    | none => rfl
    | some pair =>
      obtain ⟨q, z⟩ := pair
      exact congrArg A.trCell (h q z)
  · rfl

theorem active_cell (A : FlatAlloc) (hd : A.Disjoint) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState)
    (h : Post A f r stride M N eps p initial final) (region : RegionName) (offset : Nat)
    (hr : region ∈ A.regions) (ho : offset < A.extent region) :
    final.mem A.flat (A.addr region offset) = A.trCell
      ((Contract.expectedCell f r stride (snapshot r stride initial) M N eps p initial.pid region offset).getD (initial.mem region offset)) := by
  rw [h]
  simp only [expected, A.decode_addr hd hr ho, ↓reduceIte]

theorem frame (A : FlatAlloc) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState)
    (h : Post A f r stride M N eps p initial final)
    (region : RegionName) (offset : Nat)
    (ho : Outside A f r stride M N initial region offset) :
    final.mem region offset = (A.flattenState initial).mem region offset := by
  rw [h]
  unfold expected
  by_cases hr : region = A.flat
  · subst region
    simp only [if_pos rfl, FlatAlloc.flattenState, FlatAlloc.readFlat]
    cases hc : A.decode offset with
    | none => rfl
    | some pair =>
      obtain ⟨q, z⟩ := pair
      obtain ⟨_, hz, _⟩ := A.decode_sound hc
      have hn : ¬ Contract.written f r stride M N initial.pid q z := by
        intro hw
        rcases ho with ho | ho
        · exact ho rfl
        · exact ho q z hw hz
      change A.trCell ((Contract.expectedCell f r stride (snapshot r stride initial)
        M N eps p initial.pid q z).getD (initial.mem q z)) = A.trCell (initial.mem q z)
      rw [(Contract.expectedCell_none_iff f r stride (snapshot r stride initial)
        M N eps p initial.pid q z).mpr hn]
      rfl
  · simp only [if_neg hr, FlatAlloc.flattenState]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.FlatContract
