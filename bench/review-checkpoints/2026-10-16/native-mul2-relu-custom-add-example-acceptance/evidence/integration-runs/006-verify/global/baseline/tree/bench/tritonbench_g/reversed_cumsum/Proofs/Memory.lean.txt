import VeriTile.Triton.KernelLemmas.ScatterStore
import VeriTile.Meta.StatementAudit

/-! Full tagged-cell readback of a real masked scatter. Unlike readMem-only
readback, this records the output tag and is usable by complete cell contracts. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.Memory
open VeriTile.Triton
noncomputable section

theorem scatter_cell (region : RegionName) (shape : TileShape) (s : BlockState)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) [DecidablePred active]
    (i : TileIndex shape) (hi : active i)
    (huniq : ∀ k, active k → offset k = offset i → k = i) :
    ((TileShape.allIndices shape).foldl
      (fun u k => if active k then u.writeMem region (offset k) (value k) else u) s).mem
        region (offset i) = MemCell.real (value i) := by
  obtain ⟨before, after, he⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have hn := TileShape.allIndices_nodup shape
  rw [he, List.nodup_append, List.nodup_cons] at hn
  have hafter : i ∉ after := hn.2.1.1
  rw [he, List.foldl_append, List.foldl_cons, if_pos hi]
  rw [foldl_store_preserve_cell]
  · simp [BlockState.writeMem]
  · intro k hk ha heq
    exact hafter ((huniq k ha heq.2) ▸ hk)

#axiomsClean scatter_cell
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.Memory
