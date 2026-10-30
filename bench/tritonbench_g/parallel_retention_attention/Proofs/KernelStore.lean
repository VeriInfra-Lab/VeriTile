import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
noncomputable section

/-- Readback needs uniqueness only among active lanes. Padded lanes may form
the same integer address as another row, but never issue a store. -/
theorem masked_readback {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (indices : List ι) (state : State α) (i : ι) (member : i∈indices)
    (hi : active i) (unique : ∀ j,active j → off j=off i → j=i) :
    (indices.foldl (fun t j=>if active j then t.write r (off j) (value j) else t) state).mem
      r (off i)=value i := by
  induction indices generalizing state with
  | nil => simp at member
  | cons j rest ih =>
    rw [List.foldl_cons]
    by_cases later : i∈rest
    · exact ih _ later
    · have eq : i=j := by simpa [later] using member
      subst j
      rw [if_pos hi]
      rw [State.masked_scatter_frame r off value active rest r (off i)
        (Or.inr (by
          intro j hj activej collision
          exact later ((unique j activej collision.symm) ▸ hj)))]
      exact State.write_same _ _ _ _

def tileOffset (base D rowOffset colOffset : Nat) (i : TileIndex [h,w]) : Nat :=
  base+(rowOffset+i.1.val)*D+(colOffset+i.2.1.val)

def storeTile (region : RegionName) (base rows cols rowOffset colOffset : Nat)
    (state : State α) (values : TileIndex [h,w] → α) : State α :=
  (TileShape.allIndices [h,w]).foldl (fun t i=>
    if rowOffset+i.1.val<rows ∧ colOffset+i.2.1.val<cols then
      t.write region (tileOffset base cols rowOffset colOffset i) (.mk .real (values i))
    else t) state

theorem storeTile_read (region : RegionName) (base rows cols rowOffset colOffset : Nat)
    (state : State α) (values : TileIndex [h,w] → α) (i : TileIndex [h,w])
    (hr : rowOffset+i.1.val<rows) (hc : colOffset+i.2.1.val<cols) :
    (storeTile region base rows cols rowOffset colOffset state values).mem region
      (tileOffset base cols rowOffset colOffset i)=.mk .real (values i) := by
  apply masked_readback _ _ _ _ _ _ i (TileShape.mem_allIndices _ i) ⟨hr,hc⟩
  intro j hj eq
  have collision : (rowOffset+j.1.val)*cols+(colOffset+j.2.1.val)=
      (rowOffset+i.1.val)*cols+(colOffset+i.2.1.val) := by
    unfold tileOffset at eq
    omega
  obtain ⟨hrow,hcol⟩ := tile_injective cols (rowOffset+j.1.val) (rowOffset+i.1.val)
    (colOffset+j.2.1.val) (colOffset+i.2.1.val) hj.2 hc collision
  apply Prod.ext
  · apply Fin.ext
    omega
  · apply Prod.ext
    · apply Fin.ext
      omega
    · exact Subsingleton.elim _ _

theorem storeTile_frame (region other : RegionName) (base rows cols rowOffset colOffset : Nat)
    (state : State α) (values : TileIndex [h,w] → α) (offset : Nat) (different : other≠region) :
    (storeTile region base rows cols rowOffset colOffset state values).mem other offset=
      state.mem other offset :=
  State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl different) _

#axiomsClean masked_readback
#axiomsClean storeTile_read
#axiomsClean storeTile_frame
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
