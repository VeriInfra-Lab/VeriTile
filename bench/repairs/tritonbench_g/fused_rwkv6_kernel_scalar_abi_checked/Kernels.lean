import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.SourceParameters

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
open VeriTile.Triton
set_option maxRecDepth 40000
set_option maxHeartbeats 10000000

def signatureFP32 : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("w",.fp32),("u",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_k_h",.i32),("s_v_h",.i32),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BK","BV","USE_INITIAL_STATE","STORE_FINAL_STATE","REVERSE"]
  readPorts := ["q","k","v","w","u","h0"]
  writePorts := ["o","ht"]

def kernel (c : FusedRwkv6Kernel.Shape) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let s_k_h := c.keyPitch
  let s_v_h := c.valuePitch
  let B := c.B
  let H := c.H
  let T : Int := c.T
  let K : Int := c.K
  let V : Int := c.V
  let BK := c.BK
  let BV := c.BV
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  with_triton_call signatureFP32 call_args [s_k_h := s_k_h, s_v_h := s_v_h] in
  triton {
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

def signatureI32 : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("w",.fp32),("u",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_k_h",.i32),("s_v_h",.i32),("scale",.i32)]
  constexpr := ["B","H","T","K","V","BK","BV","USE_INITIAL_STATE","STORE_FINAL_STATE","REVERSE"]
  readPorts := ["q","k","v","w","u","h0"]
  writePorts := ["o","ht"]

def kernelI32 (c : FusedRwkv6Kernel.Shape) (scale : Int) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let s_k_h := c.keyPitch
  let s_v_h := c.valuePitch
  let B := c.B
  let H := c.H
  let T : Int := c.T
  let K : Int := c.K
  let V : Int := c.V
  let BK := c.BK
  let BV := c.BV
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  with_triton_call signatureI32 call_args [s_k_h := s_k_h, s_v_h := s_v_h, scale := scale] in
  triton {
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

def signatureI64 : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("w",.fp32),("u",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_k_h",.i32),("s_v_h",.i32),("scale",.i64)]
  constexpr := ["B","H","T","K","V","BK","BV","USE_INITIAL_STATE","STORE_FINAL_STATE","REVERSE"]
  readPorts := ["q","k","v","w","u","h0"]
  writePorts := ["o","ht"]

def kernelI64 (c : FusedRwkv6Kernel.Shape) (scale : Int) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let s_k_h := c.keyPitch
  let s_v_h := c.valuePitch
  let B := c.B
  let H := c.H
  let T : Int := c.T
  let K : Int := c.K
  let V : Int := c.V
  let BK := c.BK
  let BV := c.BV
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  with_triton_call signatureI64 call_args [s_k_h := s_k_h, s_v_h := s_v_h, scale := scale] in
  triton {
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

def signatureU64 : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("w",.fp32),("u",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_k_h",.i32),("s_v_h",.i32),("scale",.u64)]
  constexpr := ["B","H","T","K","V","BK","BV","USE_INITIAL_STATE","STORE_FINAL_STATE","REVERSE"]
  readPorts := ["q","k","v","w","u","h0"]
  writePorts := ["o","ht"]

def kernelU64 (c : FusedRwkv6Kernel.Shape) (scale : Nat) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let s_k_h := c.keyPitch
  let s_v_h := c.valuePitch
  let B := c.B
  let H := c.H
  let T : Int := c.T
  let K : Int := c.K
  let V : Int := c.V
  let BK := c.BK
  let BV := c.BV
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  with_triton_call signatureU64 call_args [s_k_h := s_k_h, s_v_h := s_v_h, scale := scale] in
  triton {
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

/-- The native raw integer-one specialization keeps the copied function body.
Float 1.0 continues to use the runtime FP32 profile. -/
def signatureOne : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("w",.fp32),("u",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_k_h",.i32),("s_v_h",.i32)]
  constexpr := ["B","H","T","K","V","BK","BV","USE_INITIAL_STATE","STORE_FINAL_STATE","REVERSE","scale"]
  readPorts := ["q","k","v","w","u","h0"]
  writePorts := ["o","ht"]

def kernelOne (c : FusedRwkv6Kernel.Shape) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let s_k_h := c.keyPitch
  let s_v_h := c.valuePitch
  let B := c.B
  let H := c.H
  let T : Int := c.T
  let K : Int := c.K
  let V : Int := c.V
  let BK := c.BK
  let BV := c.BV
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  let scale : Nat := 1
  with_triton_call signatureOne call_args [s_k_h := s_k_h, s_v_h := s_v_h] in
  triton {
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


end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
