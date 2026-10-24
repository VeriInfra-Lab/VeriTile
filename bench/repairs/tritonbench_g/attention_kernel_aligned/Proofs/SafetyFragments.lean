import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Complete

/-! Small proof terms for literal source slices. Each slice is definitionally
checked against the authoritative source. These are not alternate kernels or
runtime implementations; slicing prevents repeatedly normalizing the entire
annotated attention AST while checking one statement. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
open VeriTile Triton FP.Structural Source Full
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 2000000

def queryPrefix (p : Parameters) : List ComputeStmt :=
  let Q := p.q
  let K := p.k
  let V := p.v
  (triton {
    tl.region Q = tl.float16, K = tl.float16, V = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // $(p.heads)
    off_h = off_hz % $(p.heads)
    q_offset = off_b * $(p.qBatchStride) + off_h * $(p.qHeadStride)
    k_offset = off_b * $(p.kBatchStride) + off_h * $(p.kHeadStride)
    v_offset = off_b * $(p.vBatchStride) + off_h * $(p.vHeadStride)
    o_offset = off_b * $(p.outBatchStride) + off_h * $(p.outHeadStride)
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
      shape=($(p.length), $(p.head)), strides=($(p.qRowStride), $(p.qColStride)),
      offsets=(start_m * $(p.blockM), 0), block_shape=($(p.blockM), $(p.head)), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset,
      shape=($(p.head), $(p.length + 0)), strides=($(p.kDimStride), $(p.kSeqStride)),
      offsets=(0, 0), block_shape=($(p.head), $(p.blockN)), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset,
      shape=($(p.length + 0), $(p.head)), strides=($(p.vSeqStride), $(p.vDimStride)),
      offsets=(0, 0), block_shape=($(p.blockN), $(p.head)), order=(1, 0))
    m_i = tl.zeros([$(p.blockM)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(p.blockM)], dtype=tl.float32)
    acc = tl.zeros([$(p.blockM), $(p.head)], dtype=tl.float32)
    qk_scale = $((p.scale : ℝ)) * 1.44269504
  }).surfaceBody

theorem queryPrefix_source (p : Parameters) : queryPrefix p=(prologue p).take 15 := rfl

def queryLoad (p : Parameters) := ((prologue p).drop 15).take 1
def biasPrefix (p : Parameters) := ((prologue p).drop 16).take 6
def biasLoad (p : Parameters) := (prologue p).drop 22

#axiomsClean queryPrefix_source
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
