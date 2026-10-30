import bench.tritonbench_g.attention_fwd_triton3.Sources.SourceDefinitions
import bench.tritonbench_g.attention_fwd_triton3.Sources.Mathematical_Kernels

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Kernels
open VeriTile.Triton Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Complete selected FP32 source. Integer-width repair is represented in the
separate machine certificate; integer casts erase to mathematical indices.
No floating expression, reduction, NaN alpha guard, or store is removed. -/
def source (c : Call) : ComputeKernel :=
  FP32Source.original "Q" "K" "V" "M" "O" "L" c.scale
    c.q.batch c.q.head c.q.row c.q.dim
    c.k.batch c.k.head c.k.row c.k.dim
    c.v.batch c.v.head c.v.row c.v.dim
    c.o.batch c.o.head c.o.row c.o.dim
    c.Z c.H c.HK c.NQ c.rounded c.NK
    c.params.evenM c.params.evenN c.BM c.D c.BN c.finish c.init c.window c.complement c.offset c.size

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Kernels
