import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Complete actual Python bodies. The newly authored candidate Python only
corresponds to the existing integer head-block product candidate. -/
namespace VeriTile.Bench.Repairs.RotaryEmb.Kernels
open VeriTile.Triton

structure Layout where
  (Q K Cos Sin : RegionName)
  (stride_qbs stride_qh stride_qd stride_kbs stride_kh
    stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind
    max_total_len ROTARY_DIM HEAD_Q HEAD_K BLOCK_HEAD
    BLOCK_SEQ BLOCK_PAIRS : Nat)

def sourceSignature : CallSignature where
  pointers := [("Q",.fp32), ("K",.fp32), ("Cos",.fp32), ("Sin",.fp32)]
  scalars := [("stride_qbs",.integer), ("stride_qh",.integer), ("stride_qd",.integer), ("stride_kbs",.integer), ("stride_kh",.integer), ("stride_kd",.integer), ("stride_cosbs",.integer), ("stride_cosd",.integer), ("stride_sinbs",.integer), ("stride_sind",.integer), ("max_total_len",.integer), ("ROTARY_DIM",.integer), ("HEAD_Q",.integer), ("HEAD_K",.integer)]
  constexpr := ["BLOCK_HEAD","BLOCK_SEQ","BLOCK_PAIRS"]
  readPorts := ["Q","K","Cos","Sin"]
  writePorts := ["Q","K"]

/-- Actual selected original source. -/
def kernel (p : Layout) : ComputeKernel :=
  let Q := p.Q
  let K := p.K
  let Cos := p.Cos
  let Sin := p.Sin
  let BLOCK_HEAD := p.BLOCK_HEAD
  let BLOCK_SEQ := p.BLOCK_SEQ
  let BLOCK_PAIRS := p.BLOCK_PAIRS
  let stride_qbs : Int := p.stride_qbs
  let stride_qh : Int := p.stride_qh
  let stride_qd : Int := p.stride_qd
  let stride_kbs : Int := p.stride_kbs
  let stride_kh : Int := p.stride_kh
  let stride_kd : Int := p.stride_kd
  let stride_cosbs : Int := p.stride_cosbs
  let stride_cosd : Int := p.stride_cosd
  let stride_sinbs : Int := p.stride_sinbs
  let stride_sind : Int := p.stride_sind
  let max_total_len : Int := p.max_total_len
  let ROTARY_DIM : Int := p.ROTARY_DIM
  let HEAD_Q : Int := p.HEAD_Q
  let HEAD_K : Int := p.HEAD_K
  with_triton_call sourceSignature call_args [stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, stride_cosbs := stride_cosbs, stride_cosd := stride_cosd, stride_sinbs := stride_sinbs, stride_sind := stride_sind, max_total_len := max_total_len, ROTARY_DIM := ROTARY_DIM, HEAD_Q := HEAD_Q, HEAD_K := HEAD_K] in
  triton {
    cur_head_index = tl.program_id(0).to(tl.int64)
    cur_seq_index = tl.program_id(1).to(tl.int64)

    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD).to(tl.int64)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ).to(tl.int64)

    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64) * 2
    dim_range1 = tl.arange(0, BLOCK_PAIRS).to(tl.int64) * 2 + 1

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qd
    )

    off_dimcos_sin0 = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    off_dimcos_sin1 = cur_seq_range[:, None, None] * stride_cosbs + dim_range1[None, None, :] * stride_cosd

    off_sin0 = cur_seq_range[:, None, None] * stride_sinbs + dim_range0[None, None, :] * stride_sind
    off_sin1 = cur_seq_range[:, None, None] * stride_sinbs + dim_range1[None, None, :] * stride_sind

    q0 = tl.load(
        Q + off_q0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )

    cos0 = tl.load(Cos + off_dimcos_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin0 = tl.load(Sin + off_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    cos1 = tl.load(Cos + off_dimcos_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin1 = tl.load(Sin + off_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    out0 = q0 * cos0 - q1 * sin0
    out1 = q0 * sin1 + q1 * cos1

    tl.store(
        Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM)
    )
    tl.store(
        Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM)
    )

    off_k0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    off_dimcos_sin0 = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    off_dimcos_sin1 = cur_seq_range[:, None, None] * stride_cosbs + dim_range1[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )

    cos0 = tl.load(Cos + off_dimcos_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin0 = tl.load(Sin + off_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    cos1 = tl.load(Cos + off_dimcos_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin1 = tl.load(Sin + off_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1

    tl.store(
        K + off_k0,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
    )
    return
  }

/-- Complete pre-existing integer index candidate, with a newly authored Python counterpart. -/
def indexCommuted (p : Layout) : ComputeKernel :=
  let Q := p.Q
  let K := p.K
  let Cos := p.Cos
  let Sin := p.Sin
  let BLOCK_HEAD := p.BLOCK_HEAD
  let BLOCK_SEQ := p.BLOCK_SEQ
  let BLOCK_PAIRS := p.BLOCK_PAIRS
  let stride_qbs : Int := p.stride_qbs
  let stride_qh : Int := p.stride_qh
  let stride_qd : Int := p.stride_qd
  let stride_kbs : Int := p.stride_kbs
  let stride_kh : Int := p.stride_kh
  let stride_kd : Int := p.stride_kd
  let stride_cosbs : Int := p.stride_cosbs
  let stride_cosd : Int := p.stride_cosd
  let stride_sinbs : Int := p.stride_sinbs
  let stride_sind : Int := p.stride_sind
  let max_total_len : Int := p.max_total_len
  let ROTARY_DIM : Int := p.ROTARY_DIM
  let HEAD_Q : Int := p.HEAD_Q
  let HEAD_K : Int := p.HEAD_K
  with_triton_call sourceSignature call_args [stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, stride_cosbs := stride_cosbs, stride_cosd := stride_cosd, stride_sinbs := stride_sinbs, stride_sind := stride_sind, max_total_len := max_total_len, ROTARY_DIM := ROTARY_DIM, HEAD_Q := HEAD_Q, HEAD_K := HEAD_K] in
  triton {
    cur_head_index = tl.program_id(0).to(tl.int64)
    cur_seq_index = tl.program_id(1).to(tl.int64)

    cur_head_range = BLOCK_HEAD * cur_head_index + tl.arange(0, BLOCK_HEAD).to(tl.int64)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ).to(tl.int64)

    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64) * 2
    dim_range1 = tl.arange(0, BLOCK_PAIRS).to(tl.int64) * 2 + 1

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qd
    )

    off_dimcos_sin0 = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    off_dimcos_sin1 = cur_seq_range[:, None, None] * stride_cosbs + dim_range1[None, None, :] * stride_cosd

    off_sin0 = cur_seq_range[:, None, None] * stride_sinbs + dim_range0[None, None, :] * stride_sind
    off_sin1 = cur_seq_range[:, None, None] * stride_sinbs + dim_range1[None, None, :] * stride_sind

    q0 = tl.load(
        Q + off_q0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )

    cos0 = tl.load(Cos + off_dimcos_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin0 = tl.load(Sin + off_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    cos1 = tl.load(Cos + off_dimcos_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin1 = tl.load(Sin + off_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    out0 = q0 * cos0 - q1 * sin0
    out1 = q0 * sin1 + q1 * cos1

    tl.store(
        Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM)
    )
    tl.store(
        Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM)
    )

    off_k0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    off_dimcos_sin0 = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    off_dimcos_sin1 = cur_seq_range[:, None, None] * stride_cosbs + dim_range1[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )

    cos0 = tl.load(Cos + off_dimcos_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin0 = tl.load(Sin + off_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    cos1 = tl.load(Cos + off_dimcos_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin1 = tl.load(Sin + off_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1

    tl.store(
        K + off_k0,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
    )
    return
  }

end VeriTile.Bench.Repairs.RotaryEmb.Kernels
