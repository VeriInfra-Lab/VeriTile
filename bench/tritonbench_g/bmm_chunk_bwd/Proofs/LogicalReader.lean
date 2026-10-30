import bench.tritonbench_g.bmm_chunk_bwd.Proofs.IOResult
import bench.tritonbench_g.bmm_chunk_bwd.IO

namespace VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

theorem logical_iff (h : HostCall) (c : Tuning) :
    (logicalIO h c ⊨ fun xs q row col =>
      (∑ k : Fin h.chunk, if k.val<chunkLength h c q then xs.dout q row k * xs.a q k col else 0) +
        if h.hasResidual then xs.residual q row col else 0) ↔
    (realIO h c).Implements (mathematical h c) := by
  rfl

theorem logical_implements (h : HostCall) (choice : Fin 9)
    (distinct : RealViews.OutputDistinct (tuning choice).BM (tuning choice).BN (HostContract.parameters h))
    (separate : HostGrid.InputsSeparate (HostContract.parameters h))
    (writes : HostGrid.SeparateWrites (tuning choice).BM (tuning choice).BN
      (HostContract.parameters h) (jobs h (tuning choice))) :
    logicalIO h (tuning choice) ⊨ fun xs q row col =>
      (∑ k : Fin h.chunk, if k.val<chunkLength h (tuning choice) q then xs.dout q row k * xs.a q k col else 0) +
        if h.hasResidual then xs.residual q row col else 0 :=
  (logical_iff h (tuning choice)).mpr (implements h choice distinct separate writes)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
