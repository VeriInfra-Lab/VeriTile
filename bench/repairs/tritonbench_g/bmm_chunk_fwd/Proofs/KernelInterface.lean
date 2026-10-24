import bench.repairs.tritonbench_g.bmm_chunk_fwd.Source

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option autoImplicit false

/-- Raw dense source geometry and pointer roles; no Tensor flags or objects. -/
structure Parameters where
  shape : Host.Shape
  A : RegionName
  B : RegionName
  Out : RegionName
  Seq : Region .int
  seqBatch : Nat
  seqRow : Nat
  causal : Bool
  hasSeq : Bool

/-- Fixed FP32 specialization. The dense strides are those of the selected
kernel's admitted source buffers; no API dtype selection or copy is performed. -/
def parameters (p : Parameters) : Host.Call where
  shape := p.shape
  aStrides := Host.contiguous p.shape
  bStrides := Host.contiguous p.shape
  A := p.A
  B := p.B
  Out := p.Out
  Seq := p.Seq
  seqBatch := p.seqBatch
  seqRow := p.seqRow
  causal := p.causal
  hasSeq := p.hasSeq
end
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelInterface
