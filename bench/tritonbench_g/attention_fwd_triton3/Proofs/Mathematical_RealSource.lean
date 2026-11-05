import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Selected
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_SourceParameters
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSource
open VeriTile.Triton RealPrologue
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def source (BM BN BD Z finish : Nat) (p : Parameters) (optimized : Bool) : ComputeKernel :=
  let quote := if optimized then VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Quoted.keyMaskCSE
    else VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Quoted.kernel
  quote p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
    p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
    p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
    p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
    p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
    Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
    p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size

def kernel (BM BN BD Z finish : Nat) (p : Parameters) (optimized : Bool) : ComputeKernel :=
  (source BM BN BD Z finish p optimized).eraseDType

theorem source_projection (BM BN BD Z finish : Nat) (p : Parameters) (optimized : Bool) :
    (source BM BN BD Z finish p optimized).toAlgorithm?=.ok (source BM BN BD Z finish p optimized).toAlgKernel := by
  cases optimized <;> rfl

theorem projection (BM BN BD Z finish : Nat) (p : Parameters) (optimized : Bool) :
    (kernel BM BN BD Z finish p optimized).toAlgorithm?=.ok (kernel BM BN BD Z finish p optimized).toAlgKernel := by
  simp only [kernel,ComputeKernel.eraseDType,source_projection]
  simp [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody]

#axiomsClean source_projection
#axiomsClean projection
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSource
