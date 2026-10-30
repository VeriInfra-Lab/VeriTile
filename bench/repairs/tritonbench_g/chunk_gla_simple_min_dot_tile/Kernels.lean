import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.SourceCompatibility
import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.Layout

/-! Exact arithmetic sequence of the separate Python repair. Its pointer-index
expressions are represented as mathematical naturals; the Python source widens
all address operands to i64. The checked address lemmas are in Proofs/Address.
Program, Execution and Native connect this program to the repaired source and
prove complete execution and active native integer refinement. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
open Lean VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Storage
/-- Complete selected original FP32 source. -/
def fp32Original (out_dtype : FloatDType) (q k v h g o : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,v,h,g], outputs=[o]
    tl.region q = tl.float32, k = tl.float32, h = tl.float32, v = tl.float32, g = tl.float32, o = OUT_DTYPE
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    o_i = tl.arange(0, $(BT))
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
    b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
    rows = i_t * $(BT) + tl.arange(0, $(BT))
    cols = i_v * $(BV) + tl.arange(0, $(BV))
    for i_k in range(tl.cdiv($(K), $(BK))) {
      keys = i_k * $(BK) + tl.arange(0, $(BK))
      p_q = q + i_bh * $(s_k_h) + rows[:, None] * $(s_k_t) + keys[None, :]
      p_k = k + i_bh * $(s_k_h) + keys[:, None] + rows[None, :] * $(s_k_t)
      p_h = h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + keys[:, None] * $(s_h_t) + cols[None, :]
      b_q = tl.load(p_q, mask=(rows[:, None] < $(T)) & (keys[None, :] < $(K)), other=0)
      b_k = tl.load(p_k, mask=(keys[:, None] < $(K)) & (rows[None, :] < $(T)), other=0)
      b_h = tl.load(p_h, mask=(keys[:, None] < $(K)) & (cols[None, :] < $(V)), other=0)
      b_o += tl.dot(b_q, b_h, allow_tf32=False)
      b_s += tl.dot(b_q, b_k, allow_tf32=False)
    }
    p_g = g + i_bh * $(T) + rows
    b_g = tl.load(p_g, mask=rows < $(T), other=0)
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)
    p_v = v + i_bh * $(s_v_h) + rows[:, None] * $(s_v_t) + cols[None, :]
    b_v = tl.load(p_v, mask=(rows[:, None] < $(T)) & (cols[None, :] < $(V)), other=0)
    b_o = (b_o + tl.dot((b_s).to(tl.float32), b_v, allow_tf32=False)) * $((scale : ℝ))
    p_o = o + i_bh * $(s_v_h) + rows[:, None] * $(s_v_t) + cols[None, :]
    tl.store(p_o, (b_o).to(OUT_DTYPE), mask=(rows[:, None] < $(T)) & (cols[None, :] < $(V)))
  }

/-- Reuse the existing row-lane vector in address construction. All floating
operations and memory instruction order retain their original source order. -/
def rowReusedSource (out_dtype : FloatDType) (q k v h g o : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,v,h,g], outputs=[o]
    tl.region q = tl.float32, k = tl.float32, h = tl.float32, v = tl.float32, g = tl.float32, o = OUT_DTYPE
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    o_i = tl.arange(0, $(BT))
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
    b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
    rows = i_t * $(BT) + o_i
    cols = i_v * $(BV) + tl.arange(0, $(BV))
    for i_k in range(tl.cdiv($(K), $(BK))) {
      keys = i_k * $(BK) + tl.arange(0, $(BK))
      p_q = q + i_bh * $(s_k_h) + rows[:, None] * $(s_k_t) + keys[None, :]
      p_k = k + i_bh * $(s_k_h) + keys[:, None] + rows[None, :] * $(s_k_t)
      p_h = h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + keys[:, None] * $(s_h_t) + cols[None, :]
      b_q = tl.load(p_q, mask=(rows[:, None] < $(T)) & (keys[None, :] < $(K)), other=0)
      b_k = tl.load(p_k, mask=(keys[:, None] < $(K)) & (rows[None, :] < $(T)), other=0)
      b_h = tl.load(p_h, mask=(keys[:, None] < $(K)) & (cols[None, :] < $(V)), other=0)
      b_o += tl.dot(b_q, b_h, allow_tf32=False)
      b_s += tl.dot(b_q, b_k, allow_tf32=False)
    }
    p_g = g + i_bh * $(T) + rows
    b_g = tl.load(p_g, mask=rows < $(T), other=0)
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)
    p_v = v + i_bh * $(s_v_h) + rows[:, None] * $(s_v_t) + cols[None, :]
    b_v = tl.load(p_v, mask=(rows[:, None] < $(T)) & (cols[None, :] < $(V)), other=0)
    b_o = (b_o + tl.dot((b_s).to(tl.float32), b_v, allow_tf32=False)) * $((scale : ℝ))
    p_o = o + i_bh * $(s_v_h) + rows[:, None] * $(s_v_t) + cols[None, :]
    tl.store(p_o, (b_o).to(OUT_DTYPE), mask=(rows[:, None] < $(T)) & (cols[None, :] < $(V)))
  }

set_option maxHeartbeats 8000000 in
def source (dotStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.float32, tl.float32, scaleFirst]
      else fp32Original
  | .fp32, .fp32, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.float32, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.float32, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.float16, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.float16, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.float16, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.bfloat16, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.bfloat16, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.float32, tl.bfloat16, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.float32, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.float32, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.float32, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.float16, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.float16, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.float16, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.bfloat16, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.bfloat16, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.float16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.float32, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.float32, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.float32, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.float16, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.float16, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.float16, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float32, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float16, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then minDotSimpleGLASource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else minDotSimpleGLASource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]

def original (dotStorage valueStorage gateStorage : Storage) := source dotStorage valueStorage gateStorage Bool.false
def scaleFirst (dotStorage valueStorage gateStorage : Storage) := source dotStorage valueStorage gateStorage Bool.true


abbrev Shape := Host.Shape
/-- Candidate host specialization covers every supported Q/K/H, V and G dtype.
The output storage is independently explicit; Python chooses V's dtype. -/
def kernel (c : Shape) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) : ComputeKernel :=
  source dot value gate Bool.false out "q" "k" "v" "h" "g" "o"
    (c.T*c.K) c.K (c.T*c.V) c.V (c.NT*c.K*c.V) c.V
    scale c.T c.K c.V c.BT c.BK c.BV
/-- Same complete grid specialization, with row-lane-vector CSE only. -/
def rowReusedKernel (c : Shape) (scale : ℝ) : ComputeKernel :=
  rowReusedSource .fp32 "q" "k" "v" "h" "g" "o"
    (c.T*c.K) c.K (c.T*c.V) c.V (c.NT*c.K*c.V) c.V
    scale c.T c.K c.V c.BT c.BK c.BV

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
