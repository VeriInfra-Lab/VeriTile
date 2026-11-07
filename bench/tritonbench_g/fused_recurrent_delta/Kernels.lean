import VeriTile.Triton
import VeriTile.Triton.DSL.PythonConditionalAssignments
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Nat.Log

/-! Source geometry followed by complete repaired JIT quotations. Runtime
parameters are bound outside the copied function bodies. The proofs execute
these quotations, including in-place residual and gradient reloads. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
inductive Region where
  | q | k | v | beta | h0 | o | ht | do_ | dht
  | dq | dk | dv | dbeta | dh0 | retq | retk | retv | retbeta
  deriving DecidableEq,Repr
structure Addr where
  region : Region
  plane : Nat
  row : Nat
  col : Nat
  deriving DecidableEq,Repr
abbrev Memory (α : Type) := Addr → α
structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
def Shape.BH (s : Shape) := s.B*s.H
def Shape.BK (s : Shape) := s.K.nextPowerOfTwo
def Shape.FBV (s : Shape) := min s.V.nextPowerOfTwo 8
def Shape.BBV (s : Shape) := min s.V.nextPowerOfTwo 32
def ceil (n d : Nat) := (n+d-1)/d
def Shape.FNV (s : Shape) := ceil s.V s.FBV
def Shape.BNV (s : Shape) := ceil s.V s.BBV
def Shape.Positive (s : Shape) := 0<s.B ∧ 0<s.H ∧ 0<s.T ∧ 0<s.K ∧ 0<s.V
structure Flags where
  vectorBeta : Bool
  initial : Bool
  final : Bool
  initialGrad : Bool
  terminalGrad : Bool
  defaultBeta : Bool

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta

/-! Complete pasted source bodies. Runtime integer invocation bindings and
FP32 scale storage are outside the body. Boolean constexpr specialization
selects only source branches; no separate handwritten implementation remains. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Runtime
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

def forwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("beta",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_qk_h",.integer),("s_vo_h",.integer),("scale",.fp32),("B",.integer),("H",.integer),("T",.integer)]
  constexpr := ["K", "V", "BK", "BV", "USE_INITIAL_STATE", "STORE_FINAL_STATE", "IS_HEADWISE_BETA"]
  readPorts := ["h0", "k", "v", "q", "beta"]
  writePorts := ["v", "o", "ht"]

def backwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("beta",.fp32),("dht",.fp32),("dh0",.fp32),("do",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32),("dbeta",.fp32),("h0",.fp32)]
  scalars := [("s_qk_h",.integer),("s_vo_h",.integer),("NK",.integer),("scale",.fp32),("B",.integer),("H",.integer),("T",.integer)]
  constexpr := ["K", "V", "BK", "BV", "USE_INITIAL_STATE", "IS_HEADWISE_BETA", "USE_DH0", "USE_DHT"]
  readPorts := ["dht", "q", "k", "v", "do", "beta", "h0", "dk", "dv"]
  writePorts := ["dk", "dv", "dbeta", "dh0", "dq"]

def forwardKernel
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h,s_vo_h := s_vo_h,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _v_minus = tl.sum(h * b_k[None, :], axis=1)
        b_v -= _v_minus
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=mask_bv)
        b_v *= b_beta
        h += b_k[None, :] * b_v[:, None]
        _o = h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_beta += V if IS_HEADWISE_BETA else 1

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)
  }

def backwardKernel
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat)
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) : ComputeKernel :=
  let «do» := do_
  with_triton_call backwardSignature call_args [s_qk_h := s_qk_h,s_vo_h := s_vo_h,NK := NK,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    else:
        p_beta = beta + i_bh * T + T - 1

    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    if IS_HEADWISE_BETA:
        p_dbeta = dbeta + (i_bh + i_k * B * H + i_v * B * H * NK) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    else:
        p_dbeta = dbeta + (i_bh + i_v * B * H) * T + T - 1
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)

        d_beta = d_v * b_v if IS_HEADWISE_BETA else tl.sum(d_v * b_v)
        d_v = d_v * b_beta

        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)
        if IS_HEADWISE_BETA:
            tl.store(p_dbeta, d_beta.to(p_dbeta.dtype.element_ty), mask=mask_bv)
        else:
            tl.store(p_dbeta, d_beta.to(p_dbeta.dtype.element_ty))

        d_h -= b_k[:, None] * d_v[None, :]

        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V
        p_dbeta -= V if IS_HEADWISE_BETA else 1
        p_beta -= V if IS_HEADWISE_BETA else 1

    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h.to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])

    tl.debug_barrier()

    h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
        d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
        d_k -= tl.sum(d_v[None, :] * h, axis=1)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)

        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        h += b_k[:, None] * b_v[None, :]
        _d_q = h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += K
        p_do += V
        p_v += V
        p_dk += K
        p_dv += V
        p_dq += K
        p_beta += V if IS_HEADWISE_BETA else 1
  }

def forwardDirectOutputKernel
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h,s_vo_h := s_vo_h,B := B,H := H,T := T] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_o = o + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _v_minus = tl.sum(h * b_k[None, :], axis=1)
        b_v -= _v_minus
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=mask_bv)
        b_v *= b_beta
        h += b_k[None, :] * b_v[:, None]
        _o = h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_beta += V if IS_HEADWISE_BETA else 1

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)
  }

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Runtime
