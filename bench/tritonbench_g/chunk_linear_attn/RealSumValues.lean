import bench.tritonbench_g.chunk_linear_attn.RealGridValues
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem grid_sum_value_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (selection : LibrarySelection p) (scale : ℝ) (seed terminal : Bool)
    (hcover : p.T≤p.NT*p.BT) (hkcover : p.K≤keyBlocks p*p.BK)
    (head : Nat) (hh : head<p.B*p.H) (t : Fin p.T) (j : Fin p.V)
    (history : ReverseHistoryAt p head (t.val/p.BT) m scale terminal) :
    let g := gridMemory algebra dotPrimitive .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
    hostSumMemory algebra p selection g g "valueGradient" ((head*p.T+t.val)*p.V+j.val)=
      .mk .real ((gradients scale (inputs p head m seed)
        (fun t j => incoming p head m t.val j.val)
        (fun e j => endpoint p head m Bool.true terminal e.val j.val)).v t j) := by
  let g := gridMemory algebra dotPrimitive .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
  have ho : (head*p.T+t.val)*p.V+j.val<valueCount p := packed_lt (packed_lt hh t.isLt) j.isLt
  have hr := host_sum_read algebra p selection g g ⟨_,ho⟩
  change hostSumMemory algebra p selection g g "valueGradient" _=_
  rw [hr]
  change Cell.mk .real (sumValue algebra p selection g ⟨_,ho⟩)=_
  rw [host_sum_real]
  congr 1
  calc
    _ = ∑ kb : Fin (keyBlocks p), valuePartial p head (t.val/p.BT) m terminal scale kb.val t.val j.val := by
      apply Finset.sum_congr rfl
      intro kb _
      have ha : kb.val*valueCount p+((head*p.T+t.val)*p.V+j.val)=
          ((kb.val*(p.B*p.H)+head)*p.T+t.val)*p.V+j.val := by unfold valueCount; ring
      change (g "valuePartials" (kb.val*valueCount p+((head*p.T+t.val)*p.V+j.val))).read .real=_
      rw [ha]
      dsimp only [g]
      rw [grid_partial_real p shape m scale terminal hcover head hh t j kb history]
      rfl
    _ = _ := by
      rw [value_partials_complete p head (t.val/p.BT) m seed terminal scale t j rfl hkcover,chunkGradients_correct]

#axiomsClean grid_sum_value_real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
