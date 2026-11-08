import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.Source

/-! Complete repaired Python bodies. Storage, scalar bindings and constexpr
arguments stay outside the source quotation. The original storage interface
is retained; the public ABI selects FP32. The only candidate is the existing
integer tail-bound CSE. The former real-only dot-add reorder is removed. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 6000000
set_option linter.unusedVariables false
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16


def callSignature (dtype out_dtype : FloatDType) (hist dhist : Storage) : CallSignature where
  pointers := [("g",.fp32),("dg",.fp32),
    ("h",match hist with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),
    ("dh",match dhist with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16)]
  floatingPointers := [("q",dtype),("k",dtype),("dq",dtype),("dk",dtype),
    ("v",out_dtype),("do",out_dtype)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("s_v_h",.integer),("s_v_t",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","V","BT","BK","BV","NT"]
  readPorts := ["q","k","v","h","g","do","dh"]
  writePorts := ["dq","dk","dg"]

def originalKernel (dtype out_dtype : FloatDType) (hist dhist : Storage) (q k v h g «do» dh dq dk dg : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (callSignature dtype out_dtype hist dhist) call_args [s_k_h := s_k_h, s_k_t := s_k_t, s_v_h := s_v_h, s_v_t := s_v_t, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,), padding_option="zero")
    last_idx = min(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + i_bh * T + last_idx)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * (K * s_h_t), (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * (K * s_h_t), (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")

        b_dg_last += (tl.sum(b_h * b_dh))
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < min(BT, T-i_t*BT) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_k*n_bh + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
  }

def tailBoundKernel (dtype out_dtype : FloatDType) (hist dhist : Storage) (q k v h g «do» dh dq dk dg : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (callSignature dtype out_dtype hist dhist) call_args [s_k_h := s_k_h, s_k_t := s_k_t, s_v_h := s_v_h, s_v_t := s_v_t, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,), padding_option="zero")
    last_idx = min(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + i_bh * T + last_idx)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * (K * s_h_t), (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * (K * s_h_t), (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")

        b_dg_last += (tl.sum(b_h * b_dh))
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < (min(i_t * BT + BT, T) - i_t * BT) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_k*n_bh + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
  }

def original (qk value hist dhist : Storage)
    (q k v h g «do» dh dq dk dg : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) : ComputeKernel :=
  originalKernel qk.dtype value.dtype hist dhist q k v h g «do» dh dq dk dg s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT scale

/-- The existing selected FP32 candidate only reuses the absolute chunk end
in the final gate mask; every floating instruction stays in source order. -/
def tailBoundCSE
    (q k v h g «do» dh dq dk dg : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) : ComputeKernel :=
  tailBoundKernel .fp32 .fp32 .fp32 .fp32 q k v h g «do» dh dq dk dg s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT scale

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSource
