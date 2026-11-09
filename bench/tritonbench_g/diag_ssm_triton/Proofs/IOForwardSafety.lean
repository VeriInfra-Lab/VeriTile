import bench.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.tritonbench_g.diag_ssm_triton.Proofs.IOSafetyBase
import bench.tritonbench_g.diag_ssm_triton.Proofs.RealForwardSafety

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem forward_trace (p : ForwardLayout) (length : Nat) (sep : p.x ≠ p.y)
    (u : BlockState) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p u length bounds) :
    (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u := by
  simpa only [Kernel.TraceSafe,ForwardTransport.source,Kernels.ReadableSource.forward_preserved,Erasure.forward_body] using
    RealForwardSafety.trace_safe p length u bounds hb


end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
