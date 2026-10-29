import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatState

/-! Ordered typed stores commute with finite allocation. Active addresses
are bounded explicitly; aliases retain the source lane order. The flat
background and unspecified-value oracle are arbitrary and unchanged. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural

theorem flatten_scatter {α ι : Type} (A : FlatAlloc) (hd : A.Disjoint)
    (background : RegionName → Nat → Cell α)
    (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (active : ι → Bool) (lanes : List ι) (s : State α)
    (hb : ∀ i ∈ lanes, active i = Bool.true →
      region i ∈ A.regions ∧ offset i < A.extent (region i)) :
    flattenState A background flatUndef
      (lanes.foldl (fun t i => if active i then t.write (region i) (offset i) (value i) else t) s) =
    lanes.foldl (fun t i => if active i then
      t.write A.flat (A.addr (region i) (offset i)) (trCell A (value i)) else t)
      (flattenState A background flatUndef s) := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
      have hr : ∀ j ∈ rest, active j = Bool.true →
          region j ∈ A.regions ∧ offset j < A.extent (region j) := by
        intro j hj ha
        exact hb j (List.mem_cons_of_mem i hj) ha
      simp only [List.foldl_cons]
      cases ha : active i with
      | «false» =>
          simp only [ha, Bool.false_eq_true, if_false]
          exact ih s hr
      | «true» =>
          simp only [ha, if_true]
          rw [ih _ hr]
          obtain ⟨hmem, hbound⟩ := hb i (by simp) ha
          rw [flatten_write A hd background flatUndef s hmem hbound]

/-- Specialization to a source tile still retains collisions and lane order. -/
theorem flatten_masked_tile {α : Type} {sh : TileShape}
    (A : FlatAlloc) (hd : A.Disjoint) (background flatUndef)
    (region : TileIndex sh → RegionName) (offset : TileIndex sh → Nat)
    (value : TileIndex sh → Cell α) (active : TileIndex sh → Bool) (s : State α)
    (hb : ∀ i, active i = Bool.true →
      region i ∈ A.regions ∧ offset i < A.extent (region i)) :
    flattenState A background flatUndef
      ((TileShape.allIndices sh).foldl
        (fun t i => if active i then t.write (region i) (offset i) (value i) else t) s) =
    (TileShape.allIndices sh).foldl (fun t i => if active i then
      t.write A.flat (A.addr (region i) (offset i)) (trCell A (value i)) else t)
      (flattenState A background flatUndef s) :=
  flatten_scatter A hd background flatUndef region offset value active
    (TileShape.allIndices sh) s (fun i _ => hb i)

#axiomsClean flatten_scatter
#axiomsClean flatten_masked_tile
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
