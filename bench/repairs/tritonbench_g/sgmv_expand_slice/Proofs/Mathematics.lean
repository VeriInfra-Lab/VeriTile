import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Geometry
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open scoped BigOperators

/-- Independent ordinary matrix contraction over entry snapshots. -/
noncomputable def contraction (p : Shape) (X : Nat → Nat → ℝ)
    (W : Nat → Nat → Nat → ℝ) (adapter row column : Nat) : ℝ :=
  ∑ k ∈ Finset.range p.inner, X row k * W adapter column k

/-- The formula is defined without the kernel evaluator, grid, or host API. -/
noncomputable def mathematicalStep (p : Shape) (X : Nat → Nat → ℝ)
    (W : Nat → Nat → Nat → ℝ) (add : Bool) (s : Segment)
    (previous : Nat → Nat → ℝ) (r c : Nat) : ℝ :=
  if ActiveCell p s r c then
    let v := contraction p X W (s.adapter.getD 0) r (c-p.slice)
    if add then v + previous r c else v
  else previous r c

noncomputable def mathematicalResult (p : Shape) (X : Nat → Nat → ℝ)
    (W : Nat → Nat → Nat → ℝ) (add : Bool) (segments : List Segment)
    (entry : Nat → Nat → ℝ) :=
  segments.foldl (fun previous s => mathematicalStep p X W add s previous) entry

theorem mathematicalStep_frame (p : Shape) (X : Nat → Nat → ℝ)
    (W : Nat → Nat → Nat → ℝ) (add : Bool) (s : Segment)
    (previous : Nat → Nat → ℝ) (r c : Nat) (h : ¬ ActiveCell p s r c) :
    mathematicalStep p X W add s previous r c = previous r c := by
  simp [mathematicalStep,h]

theorem mathematicalResult_frame (p : Shape) (X : Nat → Nat → ℝ)
    (W : Nat → Nat → Nat → ℝ) (add : Bool) (segments : List Segment)
    (entry : Nat → Nat → ℝ) (r c : Nat)
    (h : ∀ s ∈ segments, ¬ ActiveCell p s r c) :
    mathematicalResult p X W add segments entry r c = entry r c := by
  induction segments generalizing entry with
  | nil => rfl
  | cons s ss ih =>
      simp only [mathematicalResult,List.foldl_cons]
      rw [show ss.foldl (fun previous s => mathematicalStep p X W add s previous)
          (mathematicalStep p X W add s entry) r c = mathematicalStep p X W add s entry r c from
        ih _ (by intro t ht; exact h t (by simp [ht]))]
      exact mathematicalStep_frame p X W add s entry r c (h s (by simp))

theorem contraction_zero_inner (p : Shape) (X : Nat → Nat → ℝ)
    (W : Nat → Nat → Nat → ℝ) (a r c : Nat) (hk : p.inner = 0) :
    contraction p X W a r c = 0 := by simp [contraction,hk]

end VeriTile.Bench.Repairs.SgmvExpandSlice
