import VeriTile.Triton.Random.DirectProofs

/-! Shape/word correspondence, not complete source or runtime validation. -/
namespace VeriTile.Bench.Tests.DirectPhiloxShapes
open VeriTile.Triton
open Random.Direct

theorem zero_round_keeps_scalar_high (c0 c2 c3 : TileIndex [0] → Random.Word)
    (c1 : Random.Word) (key : Random.Key) :
    (run (.done : Plan (uniformInputShapes [0]) 0 (uniformInputShapes [0]))
      (uniformInput c0 c2 c3 c1) key).c1 PUnit.unit = c1 := rfl

theorem ten_rounds_all_four (B : Nat) (c0 c2 c3 : TileIndex [B] → Random.Word)
    (c1 : Random.Word) (seed : Int) (i : TileIndex [B]) :
    pointwise (run (uniformPlan [B] 9) (uniformInput c0 c2 c3 c1) (Random.seedKey seed)) i =
      Random.philox 10 ⟨c0 i, c1, c2 i, c3 i⟩ (Random.seedKey seed) := by
  rw [uniformPlan_pointwise, pointwise_ofPointwise]

theorem one_round_retains_distinct_component_shapes :
    Nonempty (Plan ⟨[2, 1], [1, 3], [2, 1], [1, 3]⟩ 1
      ⟨[2, 3], [2, 1], [2, 3], [2, 1]⟩) :=
  ⟨.step (.consR (.consL .nil)) (.consR (.consL .nil)) .done⟩

#print axioms zero_rounds
#print axioms self_left
#print axioms self_right
#print axioms scalarRight_left
#print axioms scalarRight_right
#print axioms round_pointwise
#print axioms samePlan_pointwise
#print axioms uniform_first_round
#print axioms uniformPlan_pointwise
#print axioms zero_round_keeps_scalar_high
#print axioms ten_rounds_all_four
#print axioms one_round_retains_distinct_component_shapes
end VeriTile.Bench.Tests.DirectPhiloxShapes
