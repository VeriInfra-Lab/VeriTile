import VeriTile.Triton
import VeriTile.Triton.DSL.ConditionalPointerAdvances
import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Sources.Kernels

/-! Complete repaired Python bodies. Call metadata supplies FP32 storage,
the source scalar argument and constexpr launch dimensions. Both traversal
directions, optional seed/final state and each pointer update remain pasted. -/
namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

def signature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("w",.fp32),
    ("u",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_k_h",.integer),("s_v_h",.integer),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BK","BV","USE_INITIAL_STATE","STORE_FINAL_STATE","REVERSE"]
  readPorts := ["q","k","v","w","u","h0"]
  writePorts := ["o","ht"]

def originalKernel (q k v w u o h0 ht : RegionName) (s_k_h s_v_h : Nat)
    (B H T K V BK BV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [s_k_h := s_k_h,s_v_h := s_v_h] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * (T * V) + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)
  }

def candidateKernel (q k v w u o h0 ht : RegionName) (s_k_h s_v_h : Nat)
    (B H T K V BK BV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [s_k_h := s_k_h,s_v_h := s_v_h] in triton {
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * (T * V) + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)

    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)
  }

def kernel (c : FusedRwkv6Kernel.Shape) : ComputeKernel :=
  originalKernel "q" "k" "v" "w" "u" "o" "h0" "ht" c.keyPitch c.valuePitch
    c.B c.H c.T c.K c.V c.BK c.BV c.initial c.final c.reverse

def earlyUKernel (c : FusedRwkv6Kernel.Shape) : ComputeKernel :=
  candidateKernel "q" "k" "v" "w" "u" "o" "h0" "ht" c.keyPitch c.valuePitch
    c.B c.H c.T c.K c.V c.BK c.BV c.initial c.final c.reverse

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
