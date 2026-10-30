import bench.tritonbench_g.embedding_triton_kernel.IO

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Mathematics
noncomputable section

/-- Embedding lookup in the caller's weight table, or zero outside vocabulary.
Neither source execution nor a prepared/gathered input array defines this formula. -/
def formula {α : Type} (zero : α) (p : Parameters)
    (ids : IO.Ids) (weight : IO.Weights α p) (a : Fin 64) (d : Fin p.width) : α :=
  if valid : p.first ≤ ids a ∧ ids a<p.last then
    weight ⟨(ids a-p.first).toNat,by unfold IO.vocabulary; omega⟩ d
  else zero

end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Mathematics
