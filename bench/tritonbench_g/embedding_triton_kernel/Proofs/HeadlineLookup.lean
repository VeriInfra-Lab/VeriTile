import bench.tritonbench_g.embedding_triton_kernel.IO

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.HeadlineLookup

/-- The signed vocabulary test supplies the independent lookup's finite index. -/
theorem vocabulary_index (p : Parameters) (id : Int) (valid : p.first ≤ id ∧ id < p.last) :
    (id-p.first).toNat < IO.vocabulary p := by
  unfold IO.vocabulary
  omega

#axiomsClean vocabulary_index
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.HeadlineLookup
