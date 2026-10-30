import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Formula

/-! Real semantics of the source's forward inclusive scan, independently of
the loop execution. This retains every lane, including unspecified tails. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Scan
open VeriTile.Triton
open scoped BigOperators
noncomputable section

private theorem foldl_realAdd_coe (l : List ℝ) (init : ℝ) :
    (l.map (fun r => ((r : ℝ) : WithBot ℝ))).foldl WithBot.realAdd
        ((init : ℝ) : WithBot ℝ) = (((l.foldl (· + ·) init : ℝ)) : WithBot ℝ) := by
  induction l generalizing init with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.foldl_cons, WithBot.realAdd_coe_coe]
      exact ih (init + a)

private theorem scan_prefix_eq (BT : Nat) (g : Fin BT → ℝ) (L : List (Fin BT)) :
    (L.map (fun k => (some (g k) : WithBot ℝ))).foldl WithBot.realAdd
        ((0 : ℝ) : WithBot ℝ) = some ((L.map g).sum) := by
  rw [show (L.map (fun k => (some (g k) : WithBot ℝ))) =
      List.map (fun r => ((r : ℝ) : WithBot ℝ)) (L.map g) from by rw [List.map_map]; rfl]
  rw [foldl_realAdd_coe]
  congr 1
  rw [List.sum_eq_foldl]

theorem inclusive_sum (BT : Nat) (g : Fin BT → ℝ) (i : Fin BT) :
    (Tile.scan .sum ⟨0, by simp⟩ .forward
      (⟨fun idx => some (g idx.1)⟩ : Tile .real [BT])).data (i, PUnit.unit) =
        some (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), g k) := by
  rw [Tile.scan_data]
  simp only [ScanOp.eval_sum, TileShape.axisCoord, TileShape.replaceAxisCoord,
    TileShape.axisDim]
  refine Eq.trans (scan_prefix_eq BT g _) ?_
  rw [Finset.sum_map_toList]
  rfl

#axiomsClean inclusive_sum
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Scan
