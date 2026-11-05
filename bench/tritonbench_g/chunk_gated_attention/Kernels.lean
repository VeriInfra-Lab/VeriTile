import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Both repaired Python function bodies are quoted without source edits.
Runtime metadata belongs to the call signatures. The two existing candidates
change only the documented integer-address or fp32-addition operand order. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSource
open VeriTile.Triton

def cumSignature (dtype out_dtype : FloatDType) : CallSignature where
  floatingPointers := [("s",dtype),("o",out_dtype)]
  scalars := [("s_s_h",.integer),("s_s_t",.integer),("s_s_d",.integer)]
  constexpr := ["T","S","BT","BS","BH","PROGRAM_BASE"]

def cumBody (dtype out_dtype : FloatDType) (s o : RegionName)
    (T S BT BS BH PROGRAM_BASE : Nat) : ComputeKernel :=
  with_triton_signature (cumSignature dtype out_dtype) in triton {
    linear = tl.cast(tl.program_id(0), tl.int64) + PROGRAM_BASE
    NS: tl.constexpr = S // BS
    NT: tl.constexpr = T // BT
    # Some autotune configurations use fewer feature blocks than BS=16.
    # A dummy launch for an exhausted configuration has no memory effects.
    if linear < BH * NT * NS:
        i_s = linear % NS
        i_t = (linear // NS) % NT
        i_bh = linear // (NS * NT)
        o_i = tl.arange(0, BT)
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

        tile_base = i_bh * s_s_h + i_t * BT * s_s_t + i_s * BS * s_s_d
        p_s = tl.make_block_ptr(s + tile_base, (BT, BS), (s_s_t, s_s_d), (0, 0), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + tile_base, (BT, BS), (s_s_t, s_s_d), (0, 0), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def cumOriginal (dtype out_dtype : FloatDType) (s o : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS BH PROGRAM_BASE : Nat) : ComputeKernel :=
  (cumBody dtype out_dtype s o T S BT BS BH PROGRAM_BASE).withScalarArguments
    [ScalarArgument.integer "s_s_h" s_s_h, ScalarArgument.integer "s_s_t" s_s_t,
      ScalarArgument.integer "s_s_d" s_s_d]

def cumIndexFirstBody (dtype out_dtype : FloatDType) (s o : RegionName)
    (T S BT BS BH PROGRAM_BASE : Nat) : ComputeKernel :=
  with_triton_signature (cumSignature dtype out_dtype) in triton {
    linear = tl.cast(tl.program_id(0), tl.int64) + PROGRAM_BASE
    NS: tl.constexpr = S // BS
    NT: tl.constexpr = T // BT
    # Some autotune configurations use fewer feature blocks than BS=16.
    # A dummy launch for an exhausted configuration has no memory effects.
    if linear < BH * NT * NS:
        i_s = linear % NS
        i_t = (linear // NS) % NT
        i_bh = linear // (NS * NT)
        o_i = tl.arange(0, BT)
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

        tile_base = s_s_h * i_bh + i_t * BT * s_s_t + i_s * BS * s_s_d
        p_s = tl.make_block_ptr(s + tile_base, (BT, BS), (s_s_t, s_s_d), (0, 0), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + tile_base, (BT, BS), (s_s_t, s_s_d), (0, 0), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def cumIndexFirst (dtype out_dtype : FloatDType) (s o : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS BH PROGRAM_BASE : Nat) : ComputeKernel :=
  (cumIndexFirstBody dtype out_dtype s o T S BT BS BH PROGRAM_BASE).withScalarArguments
    [ScalarArgument.integer "s_s_h" s_s_h, ScalarArgument.integer "s_s_t" s_s_t,
      ScalarArgument.integer "s_s_d" s_s_d]

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def hSignature (dot gate history dtype out_dtype : FloatDType) : CallSignature where
  floatingPointers := [("k",dot),("v",dot),("g",gate),("h",history),("h0",dtype),("ht",out_dtype)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("s_k_d",.integer),
    ("s_v_h",.integer),("s_v_t",.integer),("s_v_d",.integer),
    ("s_h_h",.integer),("s_h_t",.integer),("s_h_d",.integer)]
  constexpr := ["T","K","V","BT","BK","BV","NT","GATEK","USE_INITIAL_STATE","STORE_FINAL_STATE","PROGRAM_BASE"]

def hBody (dot gate history dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  with_triton_signature (hSignature dot gate history dtype out_dtype) in triton {
    linear = tl.cast(tl.program_id(0), tl.int64) + PROGRAM_BASE
    NV: tl.constexpr = V // BV
    NK: tl.constexpr = K // BK
    i_v = linear % NV
    i_k = (linear // NV) % NK
    i_bh = linear // (NV * NK)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V + i_k * BK * V + i_v * BV, (BK, BV), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(NT, tl.int64), tl.cast(1, tl.int64)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h + i_k * BK * s_k_d + i_t * BT * s_k_t, (BK, BT), (s_k_d, s_k_t), (0, 0), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h + i_t * BT * s_v_t + i_v * BV * s_v_d, (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V + i_k * BK * s_h_t + i_v * BV * s_h_d, (BK, BV), (s_h_t, s_h_d), (0, 0), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if GATEK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h + i_k * BK * s_k_d + i_t * BT * s_k_t, (BK, BT), (s_k_d, s_k_t), (0, 0), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h + (i_t * BT + BT - 1) * s_k_t + i_k * BK * s_k_d, (BK,), (s_k_d,), (0,), (BK,), (0,))
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[:, None]
            # [BK, BT]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h + i_t * BT * s_v_t + i_v * BV * s_v_d, (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h + (i_t * BT + BT - 1) * s_v_t + i_v * BV * s_v_d, (BV,), (s_v_d,), (0,), (BV,), (0,))
            # [BV,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[None, :]
            # [BT, BV]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V + i_k * BK * V + i_v * BV, (BK, BV), (V, 1), (0, 0), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
  }

def hOriginalTyped (dot gate history dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      T K V BT BK BV NT PROGRAM_BASE : Nat)
    (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  (hBody dot gate history dtype out_dtype k v g h h0 ht T K V BT BK BV NT PROGRAM_BASE
    GATEK USE_INITIAL_STATE STORE_FINAL_STATE).withScalarArguments
    [ScalarArgument.integer "s_k_h" s_k_h,ScalarArgument.integer "s_k_t" s_k_t,
     ScalarArgument.integer "s_k_d" s_k_d,ScalarArgument.integer "s_v_h" s_v_h,
     ScalarArgument.integer "s_v_t" s_v_t,ScalarArgument.integer "s_v_d" s_v_d,
     ScalarArgument.integer "s_h_h" s_h_h,ScalarArgument.integer "s_h_t" s_h_t,
     ScalarArgument.integer "s_h_d" s_h_d]

def hSumLeftBody (dot gate history dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  with_triton_signature (hSignature dot gate history dtype out_dtype) in triton {
    linear = tl.cast(tl.program_id(0), tl.int64) + PROGRAM_BASE
    NV: tl.constexpr = V // BV
    NK: tl.constexpr = K // BK
    i_v = linear % NV
    i_k = (linear // NV) % NK
    i_bh = linear // (NV * NK)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V + i_k * BK * V + i_v * BV, (BK, BV), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(NT, tl.int64), tl.cast(1, tl.int64)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h + i_k * BK * s_k_d + i_t * BT * s_k_t, (BK, BT), (s_k_d, s_k_t), (0, 0), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h + i_t * BT * s_v_t + i_v * BV * s_v_d, (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V + i_k * BK * s_h_t + i_v * BV * s_h_d, (BK, BV), (s_h_t, s_h_d), (0, 0), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if GATEK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h + i_k * BK * s_k_d + i_t * BT * s_k_t, (BK, BT), (s_k_d, s_k_t), (0, 0), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h + (i_t * BT + BT - 1) * s_k_t + i_k * BK * s_k_d, (BK,), (s_k_d,), (0,), (BK,), (0,))
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[:, None]
            # [BK, BT]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h + i_t * BT * s_v_t + i_v * BV * s_v_d, (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h + (i_t * BT + BT - 1) * s_v_t + i_v * BV * s_v_d, (BV,), (s_v_d,), (0,), (BV,), (0,))
            # [BV,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[None, :]
            # [BT, BV]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
        # [BK, BV]
        b_h = tl.dot(b_k, b_v, allow_tf32=False) + b_h

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V + i_k * BK * V + i_v * BV, (BK, BV), (V, 1), (0, 0), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
  }
def hSumLeftTyped (dot gate history dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      T K V BT BK BV NT PROGRAM_BASE : Nat)
    (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  (hSumLeftBody dot gate history dtype out_dtype k v g h h0 ht T K V BT BK BV NT PROGRAM_BASE
    GATEK USE_INITIAL_STATE STORE_FINAL_STATE).withScalarArguments
    [ScalarArgument.integer "s_k_h" s_k_h,ScalarArgument.integer "s_k_t" s_k_t,
     ScalarArgument.integer "s_k_d" s_k_d,ScalarArgument.integer "s_v_h" s_v_h,
     ScalarArgument.integer "s_v_t" s_v_t,ScalarArgument.integer "s_v_d" s_v_d,
     ScalarArgument.integer "s_h_h" s_h_h,ScalarArgument.integer "s_h_t" s_h_t,
     ScalarArgument.integer "s_h_d" s_h_d]


def hSource (dot gate history : Storage) (swapped : Bool) :=
  if swapped then hSumLeftTyped dot.dtype gate.dtype history.dtype
  else hOriginalTyped dot.dtype gate.dtype history.dtype

def hOriginal (dot gate history : Storage) := hSource dot gate history Bool.false
def hSumLeft (dot gate history : Storage) := hSource dot gate history Bool.true

end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSource
