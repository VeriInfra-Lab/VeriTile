import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostContract
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels

namespace VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

theorem source_real (h : HostCall) (c : Tuning) :
    (Kernels.fp32Source h c).eraseDType =
      RealProgram.kernel Bool.false c.BM c.BN c.BCS (HostContract.parameters h) :=
  RealProgram.storage_erasure .fp32 .fp32 Bool.false h.residualDType h.outputDType
    c.BM c.BN c.BCS (HostContract.parameters h)

end
end VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
