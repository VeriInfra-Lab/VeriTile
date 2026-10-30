import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open VeriTile.Triton FP.Structural

/-- Clamped inactive pointers may alias. Readback requires uniqueness only
among active store destinations; reuse the existing scatter frame for the tail. -/
theorem masked_scatter_readback_active {α : Type} {shape : TileShape}
    (region : RegionName) (offset : TileIndex shape → Nat)
    (value : TileIndex shape → Cell α) (active : TileIndex shape → Prop)
    [DecidablePred active] (state : State α) (i : TileIndex shape) (hi : active i)
    (injective : ∀ i j, active i → active j → offset i = offset j → i = j) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write region (offset k) (value k) else t) state).mem
        region (offset i) = value i := by
  obtain ⟨before,after,hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have hnd := TileShape.allIndices_nodup shape
  rw [hl,List.nodup_append,List.nodup_cons] at hnd
  obtain ⟨_,⟨hnot,_⟩,_⟩ := hnd
  rw [hl,List.foldl_append,List.foldl_cons,if_pos hi,State.masked_scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj haj heq
    exact hnot ((injective i j hi haj heq) ▸ hj)

#axiomsClean masked_scatter_readback_active
end VeriTile.Bench.Repairs.SgmvExpandSlice
