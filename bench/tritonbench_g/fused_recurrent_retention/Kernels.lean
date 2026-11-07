import bench.tritonbench_g.fused_recurrent_retention.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Meta.StatementAudit

/-! Complete repaired forward/backward Python bodies and the existing exact
key-mask comparison candidate. All storage and scalar metadata stays outside
the quotations; tuple axes, loops, casts, masks and optional states remain. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Surface
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

def forwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("o",.fp32),("initial_state",.fp32),("final_state",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BK","BV","DK","DV","USE_INITIAL_STATE","STORE_FINAL_STATE"]
  readPorts := ["q","k","v","initial_state"]
  writePorts := ["o","final_state"]

def backwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32),("initial_state",.fp32),("d_final_state",.fp32)]
  scalars := [("s_do_b",.integer),("s_do_h",.integer),("s_do_t",.integer),("s_do_d",.integer),("s_df_b",.integer),("s_df_h",.integer),("s_df_k",.integer),("s_df_v",.integer),("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BK","BV","DK","DV","USE_INITIAL_STATE","USE_FINAL_GRAD"]
  readPorts := ["q","k","v","do","initial_state","d_final_state"]
  writePorts := ["dq","dk","dv"]

def forwardKernel (q k v o initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (BK BV DK DV : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h,s_qk_t := s_qk_t,s_qk_d := s_qk_d,s_vo_h := s_vo_h,s_vo_t := s_vo_t,s_vo_d := s_vo_d,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        h = b_b * h + _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)
  }

def forward (dtype out_dtype : FloatDType)
        (q k v o initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  forwardKernel q k v o initial_state final_state
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T BK BV DK DV USE_INITIAL_STATE STORE_FINAL_STATE

def backwardKernel (q k v «do» dq dk dv initial_state d_final_state : RegionName)
    (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (BK BV DK DV : Nat)
    (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
  with_triton_call backwardSignature call_args [s_do_b := s_do_b,s_do_h := s_do_h,s_do_t := s_do_t,s_do_d := s_do_d,s_df_b := s_df_b,s_df_h := s_df_h,s_df_k := s_df_k,s_df_v := s_df_v,s_qk_h := s_qk_h,s_qk_t := s_qk_t,s_qk_d := s_qk_d,s_vo_h := s_vo_h,s_vo_t := s_vo_t,s_vo_d := s_vo_d,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + (i_bh // H) * s_do_b + i_h * s_do_h + (i_v * BV + tl.arange(0, BV)) * s_do_d

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h = b_b * h + _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += DK
        p_do += s_do_t
        p_v += DV
        p_dq += DK

    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + (i_bh // H) * s_do_b + i_h * s_do_h + (i_v * BV + tl.arange(0, BV)) * s_do_d + (T - 1) * s_do_t
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + (T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRAD:
        p_df = d_final_state + (i_bh // H) * s_df_b + i_h * s_df_h + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * s_df_k + \
            (i_v * BV + tl.arange(0, BV)[None, :]) * s_df_v
        d_h = tl.load(p_df, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)

        d_h *= b_b
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do -= s_do_t
        p_q -= DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV
  }

def backward (dtype out_dtype : FloatDType)
        (q k v do_ dq dk dv initial_state d_final_state : RegionName)
        (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v : Nat)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
  backwardKernel q k v do_ dq dk dv initial_state d_final_state
    s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T BK BV DK DV USE_INITIAL_STATE USE_FINAL_GRAD

def forwardCompareRightKernel (q k v o initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (BK BV DK DV : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h,s_qk_t := s_qk_t,s_qk_d := s_qk_d,s_vo_h := s_vo_h,s_vo_t := s_vo_t,s_vo_d := s_vo_d,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = DK > (i_k * BK + tl.arange(0, BK))
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        h = b_b * h + _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)
  }

def forwardCompareRight (dtype out_dtype : FloatDType)
        (q k v o initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  forwardCompareRightKernel q k v o initial_state final_state
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T BK BV DK DV USE_INITIAL_STATE STORE_FINAL_STATE

def backwardCompareRightKernel (q k v «do» dq dk dv initial_state d_final_state : RegionName)
    (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (BK BV DK DV : Nat)
    (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
  with_triton_call backwardSignature call_args [s_do_b := s_do_b,s_do_h := s_do_h,s_do_t := s_do_t,s_do_d := s_do_d,s_df_b := s_df_b,s_df_h := s_df_h,s_df_k := s_df_k,s_df_v := s_df_v,s_qk_h := s_qk_h,s_qk_t := s_qk_t,s_qk_d := s_qk_d,s_vo_h := s_vo_h,s_vo_t := s_vo_t,s_vo_d := s_vo_d,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + (i_bh // H) * s_do_b + i_h * s_do_h + (i_v * BV + tl.arange(0, BV)) * s_do_d

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = DK > (i_k * BK + tl.arange(0, BK))
    mask_bv = i_v * BV + tl.arange(0, BV) < DV

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h = b_b * h + _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += DK
        p_do += s_do_t
        p_v += DV
        p_dq += DK

    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + (i_bh // H) * s_do_b + i_h * s_do_h + (i_v * BV + tl.arange(0, BV)) * s_do_d + (T - 1) * s_do_t
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + (T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRAD:
        p_df = d_final_state + (i_bh // H) * s_df_b + i_h * s_df_h + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * s_df_k + \
            (i_v * BV + tl.arange(0, BV)[None, :]) * s_df_v
        d_h = tl.load(p_df, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)

        d_h *= b_b
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do -= s_do_t
        p_q -= DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV
  }

def backwardCompareRight (dtype out_dtype : FloatDType)
        (q k v do_ dq dk dv initial_state d_final_state : RegionName)
        (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v : Nat)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
  backwardCompareRightKernel q k v do_ dq dk dv initial_state d_final_state
    s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T BK BV DK DV USE_INITIAL_STATE USE_FINAL_GRAD

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Surface
