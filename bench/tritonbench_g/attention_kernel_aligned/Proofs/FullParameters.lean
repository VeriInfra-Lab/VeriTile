import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPBackend
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.Full
open VeriTile.Triton FP.Structural
noncomputable section

/-- Every source argument used by the original wrapper, including the two
ignored head strides. Shape/stride restrictions are not silently added here.
The wrapper asserts P_SEQ = 0, so `length` is the original N_CTX. -/
structure Parameters where
  q : RegionName
  k : RegionName
  v : RegionName
  bias : RegionName
  output : RegionName
  scale : ℝ
  qHeadStride : Nat
  qRowStride : Nat
  qColStride : Nat
  kHeadStride : Nat
  kSeqStride : Nat
  kDimStride : Nat
  vHeadStride : Nat
  vSeqStride : Nat
  vDimStride : Nat
  outHeadStride : Nat
  outRowStride : Nat
  outColStride : Nat
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

/-- `range(0, N_CTX, BLOCK_N)` visits the final partial block in full. -/
def Parameters.blocks (p : Parameters) : Nat :=
  (p.length + p.blockN - 1) / p.blockN

theorem block_guard (p : Parameters) (c : Nat) (hBN : 0 < p.blockN) :
    c * p.blockN < p.length ↔ c < p.blocks := by
  unfold Parameters.blocks
  rw [Nat.lt_div_iff_mul_lt hBN]
  have hs : (p.length + p.blockN - 1) - (p.blockN - 1) = p.length := by omega
  rw [hs]

#axiomsClean block_guard
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.Full
