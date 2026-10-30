import bench.tritonbench_g.seeded_dropout.Proofs.Memory
import VeriTile.Triton.Memory.Flatten

/-! Finite-allocation contracts for the proved writeback phase. These lemmas
are prepared for the complete execution proof; they do not assume or assert
that a whole kernel ran. Output tags, pointer-cell relocation and unallocated
flat addresses remain visible in the full-cell frame. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.FlatMemory
open VeriTile.Triton
noncomputable section

structure Bounds (x y : RegionName) (N B : Nat) (u : BlockState)
    (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, Index.offset u.pid B j < N → Index.offset u.pid B j < bounds x
  output : ∀ j : Fin B, Index.offset u.pid B j < N → Index.offset u.pid B j < bounds y

theorem bounds_of_extents (x y : RegionName) (N B : Nat) (u : BlockState)
    (bounds : RegionBounds) (hx : N ≤ bounds x) (hy : N ≤ bounds y) :
    Bounds x y N B u bounds :=
  ⟨fun _ hj => lt_of_lt_of_le hj hx, fun _ hj => lt_of_lt_of_le hj hy⟩

def Outside (A : FlatAlloc) (initial : BlockState) (y : RegionName)
    (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ A.flat ∨ ∀ j : Fin B, Index.offset initial.pid B j < N →
    o ≠ A.addr y (Index.offset initial.pid B j)

theorem storeBlock_active_cell (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (initial base : BlockState) (x y : RegionName) (N B : Nat) (p : ℝ) (seed : Int)
    (hb : Bounds x y N B initial A.extent) (j : Fin B)
    (hj : Index.offset initial.pid B j < N) :
    (A.flattenState (Memory.storeBlock initial base x y N B p seed)).mem A.flat
      (A.addr y (Index.offset initial.pid B j)) =
        MemCell.real (Formula.expected (initial.readMem x) p seed (Index.offset initial.pid B j)) := by
  have hy := hb.output j hj
  have hr : y ∈ A.regions := by
    by_contra hn
    rw [hc y hn] at hy
    omega
  rw [A.flattenState_mem_addr hd _ hr hy, Memory.storeBlock_active_cell _ _ _ _ _ _ _ _ _ hj]
  rfl

theorem storeBlock_frame (A : FlatAlloc) (initial base : BlockState)
    (x y : RegionName) (N B : Nat) (p : ℝ) (seed : Int)
    (r : RegionName) (o : Nat) (ho : Outside A initial y N B r o) :
    (A.flattenState (Memory.storeBlock initial base x y N B p seed)).mem r o =
      (A.flattenState base).mem r o := by
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState]
    unfold FlatAlloc.readFlat
    cases hc : A.decode o with
    | none => rfl
    | some pair =>
      obtain ⟨q, z⟩ := pair
      obtain ⟨_, hz, _⟩ := A.decode_sound hc
      apply congrArg A.trCell
      apply Memory.storeBlock_frame
      intro j hj ⟨hq, hoz⟩
      subst q
      rcases ho with hn | hn
      · exact hn rfl
      · exact hn j hj (by rw [hoz]; exact hz)
  · simp only [FlatAlloc.flattenState, if_neg hr]

theorem storeBlock_readback (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (initial base : BlockState) (x y : RegionName) (N B : Nat) (p : ℝ) (seed : Int)
    (hb : Bounds x y N B initial A.extent) (j : Fin B)
    (hj : Index.offset initial.pid B j < N) :
    (A.flattenState (Memory.storeBlock initial base x y N B p seed)).readMem A.flat
      (A.addr y (Index.offset initial.pid B j)) =
        Formula.expected (initial.readMem x) p seed (Index.offset initial.pid B j) := by
  have h := storeBlock_active_cell A hd hc initial base x y N B p seed hb j hj
  simp only [BlockState.readMem, h]
  rfl

end
end VeriTile.Bench.TritonBenchG.SeededDropout.FlatMemory
