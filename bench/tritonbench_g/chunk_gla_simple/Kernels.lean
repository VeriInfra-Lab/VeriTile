import bench.tritonbench_g.chunk_gla_simple.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.TupleScalarCasts
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete repaired source and its existing row-vector reuse candidate.
Storage, runtime arguments and constexpr shapes are supplied by the caller;
all source casts, tuple assignments, loop bounds, masks and dots stay visible. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 12000000
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Storage

def callSignature (out_dtype : FloatDType) : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("h",.fp32),("g",.fp32)]
  floatingPointers := [("o",out_dtype)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("s_v_h",.integer),("s_v_t",.integer),
    ("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","V","BT","BK","BV"]
  readPorts := ["q","k","v","h","g"]
  writePorts := ["o"]

def originalKernel (out_dtype : FloatDType) (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) : ComputeKernel :=
  with_triton_call (callSignature out_dtype) call_args [s_k_h := s_k_h,s_k_t := s_k_t,
    s_v_h := s_v_h,s_v_t := s_v_t,s_h_h := s_h_h,s_h_t := s_h_t,scale := scale] in triton {
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    s_k_h, s_k_t = tl.cast(s_k_h, tl.int64), tl.cast(s_k_t, tl.int64)
    s_v_h, s_v_t = tl.cast(s_v_h, tl.int64), tl.cast(s_v_t, tl.int64)
    s_h_h, s_h_t = tl.cast(s_h_h, tl.int64), tl.cast(s_h_t, tl.int64)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    rows = i_t * BT + tl.arange(0, BT).to(tl.int64)
    cols = i_v * BV + tl.arange(0, BV).to(tl.int64)
    # Keep range's signed induction bound valid across the uint32 constant interval.
    for i_k in range(tl.cast(tl.cdiv(K, BK), tl.int64)):
        keys = tl.cast(i_k, tl.int64) * BK + tl.arange(0, BK).to(tl.int64)
        p_q = q + i_bh * s_k_h + rows[:, None] * s_k_t + keys[None, :]
        p_k = k + i_bh * s_k_h + keys[:, None] + rows[None, :] * s_k_t
        p_h = h + i_bh * s_h_h + i_t * K * V + keys[:, None] * s_h_t + cols[None, :]
        b_q = tl.load(p_q, mask=(rows[:, None] < T) & (keys[None, :] < K), other=0)
        b_k = tl.load(p_k, mask=(keys[:, None] < K) & (rows[None, :] < T), other=0)
        b_h = tl.load(p_h, mask=(keys[:, None] < K) & (cols[None, :] < V), other=0)
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = g + i_bh * T + rows
    b_g = tl.load(p_g, mask=rows < T, other=0)
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    p_v = v + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    b_v = tl.load(p_v, mask=(rows[:, None] < T) & (cols[None, :] < V), other=0)
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = o + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < V))
  }

def rowReusedBodyPastedKernel (out_dtype : FloatDType) (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) : ComputeKernel :=
  with_triton_call (callSignature out_dtype) call_args [s_k_h := s_k_h,s_k_t := s_k_t,
    s_v_h := s_v_h,s_v_t := s_v_t,s_h_h := s_h_h,s_h_t := s_h_t,scale := scale] in triton {
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    s_k_h, s_k_t = tl.cast(s_k_h, tl.int64), tl.cast(s_k_t, tl.int64)
    s_v_h, s_v_t = tl.cast(s_v_h, tl.int64), tl.cast(s_v_t, tl.int64)
    s_h_h, s_h_t = tl.cast(s_h_h, tl.int64), tl.cast(s_h_t, tl.int64)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    rows = i_t * BT + o_i.to(tl.int64)
    cols = i_v * BV + tl.arange(0, BV).to(tl.int64)
    # Keep range's signed induction bound valid across the uint32 constant interval.
    for i_k in range(tl.cast(tl.cdiv(K, BK), tl.int64)):
        keys = tl.cast(i_k, tl.int64) * BK + tl.arange(0, BK).to(tl.int64)
        p_q = q + i_bh * s_k_h + rows[:, None] * s_k_t + keys[None, :]
        p_k = k + i_bh * s_k_h + keys[:, None] + rows[None, :] * s_k_t
        p_h = h + i_bh * s_h_h + i_t * K * V + keys[:, None] * s_h_t + cols[None, :]
        b_q = tl.load(p_q, mask=(rows[:, None] < T) & (keys[None, :] < K), other=0)
        b_k = tl.load(p_k, mask=(keys[:, None] < K) & (rows[None, :] < T), other=0)
        b_h = tl.load(p_h, mask=(keys[:, None] < K) & (cols[None, :] < V), other=0)
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = g + i_bh * T + rows
    b_g = tl.load(p_g, mask=rows < T, other=0)
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    p_v = v + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    b_v = tl.load(p_v, mask=(rows[:, None] < T) & (cols[None, :] < V), other=0)
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = o + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < V))
  }

def fp32Original (out_dtype : FloatDType) (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) : ComputeKernel :=
  originalKernel out_dtype q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV

def rowReusedSource (out_dtype : FloatDType) (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) : ComputeKernel :=
  rowReusedBodyPastedKernel out_dtype q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV

abbrev Shape := Host.Shape
/-- The repair wrapper admits FP32 tensors. Its complete three-dimensional
launch selects this shared source, including every K block and masked tail. -/
def kernel (c : Shape) (scale : ℝ) : ComputeKernel :=
  fp32Original .fp32 "q" "k" "v" "h" "g" "o"
    (c.T*c.K) c.K (c.T*c.V) c.V (c.NT*c.K*c.V) c.V
    scale c.T c.K c.V c.BT c.BK c.BV
/-- Same complete grid specialization, with row-lane-vector CSE only. -/
def rowReusedKernel (c : Shape) (scale : ℝ) : ComputeKernel :=
  rowReusedSource .fp32 "q" "k" "v" "h" "g" "o"
    (c.T*c.K) c.K (c.T*c.V) c.V (c.NT*c.K*c.V) c.V
    scale c.T c.K c.V c.BT c.BK c.BV

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
