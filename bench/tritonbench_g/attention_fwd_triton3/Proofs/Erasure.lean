import bench.tritonbench_g.attention_fwd_triton3.Sources.Kernels
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Host

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Kernels
open VeriTile.Triton Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem real_erasure (c : Call) : (source c).eraseDType.toAlgKernel=c.source := by
  have h:=FP32Profile.erases "Q" "K" "V" "M" "O" "L" c.scale
    c.q.batch c.q.head c.q.row c.q.dim
    c.k.batch c.k.head c.k.row c.k.dim
    c.v.batch c.v.head c.v.row c.v.dim
    c.o.batch c.o.head c.o.row c.o.dim
    c.Z c.H c.HK c.NQ c.rounded c.NK
    c.params.evenM c.params.evenN c.BM c.D c.BN c.finish c.init c.window c.complement c.offset c.size
  exact congrArg ComputeKernel.toAlgKernel h

#print axioms real_erasure
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Kernels
