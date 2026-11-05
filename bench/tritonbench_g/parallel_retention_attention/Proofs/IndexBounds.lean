import bench.tritonbench_g.parallel_attention.Proofs.Sources.Mathematical_FPForwardGraph
import bench.tritonbench_g.parallel_retention_attention.Proofs.Index32

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (ForwardConfig queryChunk keyChunk valueChunk)

/-- Bounds for the original source int32 block-pointer coordinates. These are
internal execution facts; public launches must derive them from the Python
wrapper's existing ShapeContract/IndexDomain and actual program coordinate. -/
structure ForwardIndexBounds (c : ForwardConfig) (s : State α) : Prop where
  time : queryChunk c s+c.BTL+c.BTS≤2147483648
  key : keyChunk c s+c.BK≤2147483648
  value : valueChunk c s+c.BV≤2147483648

namespace ForwardIndexBounds
 theorem lanes (h : ForwardIndexBounds c s) : c.BTL≤2147483648 := by
  have := h.time
  omega

 theorem query_lt (h : ForwardIndexBounds c s) : queryChunk c s<2147483648 := by
  by_cases zero : c.BTL=0
  · simp [queryChunk,zero]
  · have := h.time
    omega
 theorem key_lt (h : ForwardIndexBounds c s) : keyChunk c s<2147483648 := by
  by_cases zero : c.BK=0
  · simp [keyChunk,zero]
  · have := h.key
    omega
 theorem value_lt (h : ForwardIndexBounds c s) : valueChunk c s<2147483648 := by
  by_cases zero : c.BV=0
  · simp [valueChunk,zero]
  · have := h.value
    omega

 theorem query_cast (h : ForwardIndexBounds c s) :
     Index32.signed ((s.pids 1 : Int)*c.BTL)=(queryChunk c s : Int) := by
   simpa only [queryChunk,Int.natCast_mul] using Index32.nat_value (queryChunk c s) h.query_lt
 theorem key_cast (h : ForwardIndexBounds c s) :
     Index32.signed (((s.pids 0 : Int)/(((c.V+c.BV-1)/c.BV : Nat) : Int))*c.BK)=(keyChunk c s : Int) := by
   simpa only [keyChunk,Int.natCast_mul,Int.natCast_ediv] using Index32.nat_value (keyChunk c s) h.key_lt
 theorem value_cast (h : ForwardIndexBounds c s) :
     Index32.signed (((s.pids 0 % ((c.V+c.BV-1)/c.BV) : Nat) : Int)*c.BV)=(valueChunk c s : Int) := by
   simpa only [valueChunk,Int.natCast_mul] using Index32.nat_value (valueChunk c s) h.value_lt

 theorem key_cast_eval (h : ForwardIndexBounds c s) :
     Index32.signed (((s.pids 0 : Int)/((c.V+c.BV-1 : Nat)/ (c.BV : Int)))*c.BK)=(keyChunk c s : Int) := by
   simpa only [Int.natCast_ediv] using h.key_cast

 theorem value_cast_eval (h : ForwardIndexBounds c s) :
     Index32.signed (((s.pids 0 : Int)%((c.V+c.BV-1 : Nat)/(c.BV : Int)))*c.BV)=(valueChunk c s : Int) := by
   simpa only [Int.natCast_emod,Int.natCast_ediv] using h.value_cast
end ForwardIndexBounds
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
