import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Triton FP.Structural
noncomputable section

/-- Repaired independent logical view metadata. The public wrapper selects
standard power-of-two widths and P_SEQ=0; both precision branches share this metadata.
The final four fields are integer-only launch and decoded compiler-layout metadata. -/
structure Parameters where
  q : RegionName
  k : RegionName
  v : RegionName
  bias : RegionName
  output : RegionName
  scale : ℝ
  /-- Exact observed FP32 launch word, including signed zero. -/
  scaleWord : BitVec 32
  qBatchStride : Nat
  qHeadStride : Nat
  qRowStride : Nat
  qColStride : Nat
  kBatchStride : Nat
  kHeadStride : Nat
  kSeqStride : Nat
  kDimStride : Nat
  vBatchStride : Nat
  vHeadStride : Nat
  vSeqStride : Nat
  vDimStride : Nat
  outBatchStride : Nat
  outHeadStride : Nat
  outRowStride : Nat
  outColStride : Nat
  biasBatchStride : Nat
  biasHeadStride : Nat
  biasRowStride : Nat
  batches : Nat
  heads : Nat
  length : Nat
  blockM : Nat
  blockN : Nat
  head : Nat
  biasLast : Nat
  biasNumel : Nat
  numWarps : Nat := 4
  numStages : Nat := 2
  nativeColumns : Nat := 1
  nativeSerial : Bool := Bool.false

/-- `range(0, N_CTX, BLOCK_N)` visits the final partial block in full. -/
def Parameters.blocks (p : Parameters) : Nat :=
  (p.length + p.blockN - 1) / p.blockN

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Full
