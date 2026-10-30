import bench.tritonbench_g.destindex_copy_kv2.Proofs.Tile

namespace VeriTile.Bench.Repairs.DestindexCopyKv2
open VeriTile.Triton FP.Structural
noncomputable section

/-- The omitted feature mask is true for every actual arange lane. -/
theorem same_source_memory {α : Type} [Inhabited α] (rest : Algebra α) (p : Layout)
    (width : p.head_dim=p.BLOCK_DMODEL) (s : State α) :
    (Tile.execution rest p s).val.mem=(DestindexCopyKv1.Tile.execution rest p s).val.mem := by
  rw [Tile.execution_memory,DestindexCopyKv1.Tile.execution_memory]
  have equal : Tile.copied p s=DestindexCopyKv1.Tile.copied p s := by
    unfold Tile.copied DestindexCopyKv1.Tile.copied
    congr 1
    funext t i
    have dim : i.2.1.val<p.head_dim := by rw [width]; exact i.2.1.isLt
    simp only [Tile.active,DestindexCopyKv1.Tile.active,dim,and_true]
    rfl
  rw [equal]

#axiomsClean same_source_memory
end
end VeriTile.Bench.Repairs.DestindexCopyKv2
