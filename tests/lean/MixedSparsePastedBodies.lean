import bench.tritonbench_g.mixed_sparse_attention.Proofs.MixedSparseAttention_Source_Config
namespace VeriTile.Tests.MixedSparsePastedBodies
open VeriTile.Triton
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source

/-- The complete source retains its original ordered port list, including
both K/V occurrences. Metadata views do not reorder or deduplicate inputs. -/
example (c : Config) : (parsedKernel c).inputs=
    [Region.cast c.seq,Region.cast c.blocks,Region.cast c.counts,c.q,
      Region.cast c.offsets,c.k,c.v,Region.cast c.columns,c.k,c.v] ∧
    (parsedKernel c).outputs=[c.out] := by
  cases hs : c.storage <;> simp only [parsedKernel,hs] <;> exact ⟨rfl,rfl⟩
end VeriTile.Tests.MixedSparsePastedBodies
