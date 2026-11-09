import bench.tritonbench_g.parallel_retention_attention.Proofs.Index32
import bench.tritonbench_g.parallel_attention.Proofs.Sources.Mathematical_FPBackwardGraph

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)

/-- Internal int32-coordinate bounds for the source's DQ and DK/DV helpers. -/
structure HelperIndexBounds (c : BackwardConfig) : Prop where
  time : c.i_c*c.BTL+c.BTL+c.BTS≤2147483648
  key : c.i_k*c.BK+c.BK≤2147483648
  value : c.i_v*c.BV+c.BV≤2147483648

namespace HelperIndexBounds
theorem lanes (h : HelperIndexBounds c) : c.BTL≤2147483648 := by
  have := h.time
  omega

theorem query_lt (h : HelperIndexBounds c) : c.i_c*c.BTL<2147483648 := by
  by_cases zero : c.BTL=0
  · simp [zero]
  · have := h.time
    omega

theorem key_lt (h : HelperIndexBounds c) : c.i_k*c.BK<2147483648 := by
  by_cases zero : c.BK=0
  · simp [zero]
  · have := h.key
    omega

theorem value_lt (h : HelperIndexBounds c) : c.i_v*c.BV<2147483648 := by
  by_cases zero : c.BV=0
  · simp [zero]
  · have := h.value
    omega

theorem query_cast (h : HelperIndexBounds c) :
    Index32.signed ((c.i_c : Int)*c.BTL)=((c.i_c*c.BTL : Nat) : Int) := by
  simpa only [Int.natCast_mul] using Index32.nat_value (c.i_c*c.BTL) h.query_lt

theorem key_cast (h : HelperIndexBounds c) :
    Index32.signed ((c.i_k : Int)*c.BK)=((c.i_k*c.BK : Nat) : Int) := by
  simpa only [Int.natCast_mul] using Index32.nat_value (c.i_k*c.BK) h.key_lt

theorem value_cast (h : HelperIndexBounds c) :
    Index32.signed ((c.i_v : Int)*c.BV)=((c.i_v*c.BV : Nat) : Int) := by
  simpa only [Int.natCast_mul] using Index32.nat_value (c.i_v*c.BV) h.value_lt
end HelperIndexBounds

#axiomsClean HelperIndexBounds.query_cast
#axiomsClean HelperIndexBounds.key_cast
#axiomsClean HelperIndexBounds.value_cast
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
