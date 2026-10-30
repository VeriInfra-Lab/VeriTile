import bench.repairs.tritonbench_g.matrix_vector_multip.SourceDomain

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelConfig
open VeriTile.Triton

theorem rowShape (c : KernelConfig) : Random.ArangeSizeDomain c.blockRows := by
  have h := c.rows_mem
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h | h | h | h <;> rw [h] <;> norm_num [Random.ArangeSizeDomain]
  all_goals first | exact ⟨0,rfl⟩ | exact ⟨1,rfl⟩ | exact ⟨2,rfl⟩ | exact ⟨3,rfl⟩

theorem columnShape (c : KernelConfig) : Random.ArangeSizeDomain c.blockColumns := by
  have h := c.columns_mem
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h | h | h <;> rw [h] <;> norm_num [Random.ArangeSizeDomain]
  all_goals first | exact ⟨5,rfl⟩ | exact ⟨6,rfl⟩ | exact ⟨7,rfl⟩

end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelConfig
