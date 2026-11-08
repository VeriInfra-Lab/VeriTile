import bench.tritonbench_g.fast_rope_embedding.Kernels

/-! Compatibility names for all retained storage combinations. Every profile
uses the two literal Python quotations in the authoritative Kernels module. -/
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.Repairs.FastRopeEmbedding.Kernels.Storage

def rope (input cosine sine : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.FastRopeEmbedding.Kernels.commutedForKernel input cosine sine
  else VeriTile.Bench.Repairs.FastRopeEmbedding.Kernels.ropeForKernel input cosine sine

end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource
