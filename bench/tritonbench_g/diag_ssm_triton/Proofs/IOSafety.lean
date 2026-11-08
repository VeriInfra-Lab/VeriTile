import bench.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.tritonbench_g.diag_ssm_triton.Proofs.IOForwardSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.IOBackwardSafetyBase
import bench.tritonbench_g.diag_ssm_triton.Proofs.RealBackwardSafety

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem backward_trace (p : BackwardLayout) (length : Nat)
    (sep : RegionCorrectness.BackwardSeparation p) (u : BlockState) (bounds : RegionBounds)
    (hb : BackwardIO.Bounds p u length bounds) :
    (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u := by
  simpa only [Kernel.TraceSafe,BackwardTransport.source,Kernels.ReadableSource.backward_preserved,Erasure.backward_body] using
    RealBackwardSafety.trace_safe p length u bounds hb

#axiomsClean forward_trace
#axiomsClean backward_trace

end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
