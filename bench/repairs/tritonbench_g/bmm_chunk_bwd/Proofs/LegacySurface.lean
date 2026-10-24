import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels
namespace VeriTile.Bench.Repairs.BmmChunkBwd.LegacySurface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
def bindMain (version : Bool) (h : HostCall) (c : Tuning) : ComputeKernel :=
  (if version then bmmChunkBwdSource%[tl.float32, tl.float32, tl.float32, indexCommuted] else bmmChunkBwdSource%[tl.float32, tl.float32, tl.float32, original])
    h.residualDType h.outputDType h.A h.Dout h.Out h.Residual h.seqlen h.chunk h.K h.groupCount
    h.aStrides.batch h.aStrides.sequence (h.headStride h.aStrides.head) h.aStrides.feature
    h.doutStrides.batch h.doutStrides.chunk (h.headStride h.doutStrides.head) h.doutStrides.row h.doutStrides.column
    h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature
    (h.resStride h.residualStrides.batch) (h.resStride h.residualStrides.sequence)
    (h.resStride (h.headStride h.residualStrides.head)) (h.resStride h.residualStrides.feature)
    c.BM c.BN c.BCS h.hasResidual

def fp32Source := bindMain Bool.false
def indexCommuted := bindMain Bool.true

end VeriTile.Bench.Repairs.BmmChunkBwd.LegacySurface
