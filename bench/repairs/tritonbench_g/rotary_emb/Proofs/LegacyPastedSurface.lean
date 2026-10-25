import bench.repairs.tritonbench_g.rotary_emb.Kernels
namespace VeriTile.Bench.Repairs.RotaryEmb.Kernels.LegacyPastedSurface
open VeriTile.Triton

/-- Complete transcription of `_rotary_kernel`; FP32 payloads and signed int64 address arithmetic. -/
def kernel (p : Layout) : ComputeKernel :=
  let Q := p.Q
  let K := p.K
  let Cos := p.Cos
  let Sin := p.Sin
  triton {
    tl.region Q = tl.float32, K = tl.float32, Cos = tl.float32, Sin = tl.float32
    cur_head_index = tl.program_id(0).to(tl.int64)
    cur_seq_index = tl.program_id(1).to(tl.int64)
    cur_head_range = cur_head_index * $((p.BLOCK_HEAD : Int)) + tl.arange(0, $(p.BLOCK_HEAD)).to(tl.int64)
    cur_seq_range = cur_seq_index * $((p.BLOCK_SEQ : Int)) + tl.arange(0, $(p.BLOCK_SEQ)).to(tl.int64)
    dim_range0 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64) * $((2 : Int))
    dim_range1 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64) * $((2 : Int)) + $((1 : Int))
    off_q0 = cur_seq_range[:, None, None] * $((p.stride_qbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_qh : Int)) +
      dim_range0[None, None, :] * $((p.stride_qd : Int))
    off_q1 = cur_seq_range[:, None, None] * $((p.stride_qbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_qh : Int)) +
      dim_range1[None, None, :] * $((p.stride_qd : Int))
    off_dimcos_sin0 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range0[None, None, :] * $((p.stride_cosd : Int))
    off_dimcos_sin1 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range1[None, None, :] * $((p.stride_cosd : Int))
    off_sin0 = cur_seq_range[:, None, None] * $((p.stride_sinbs : Int)) +
      dim_range0[None, None, :] * $((p.stride_sind : Int))
    off_sin1 = cur_seq_range[:, None, None] * $((p.stride_sinbs : Int)) +
      dim_range1[None, None, :] * $((p.stride_sind : Int))
    q0 = tl.load(Q + off_q0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    q1 = tl.load(Q + off_q1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos0 = tl.load(Cos + off_dimcos_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin0 = tl.load(Sin + off_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos1 = tl.load(Cos + off_dimcos_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin1 = tl.load(Sin + off_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    out0 = q0 * cos0 - q1 * sin0
    out1 = q0 * sin1 + q1 * cos1
    tl.store(Q + off_q0, out0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
    tl.store(Q + off_q1, out1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
    off_k0 = cur_seq_range[:, None, None] * $((p.stride_kbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_kh : Int)) +
      dim_range0[None, None, :] * $((p.stride_kd : Int))
    off_k1 = cur_seq_range[:, None, None] * $((p.stride_kbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_kh : Int)) +
      dim_range1[None, None, :] * $((p.stride_kd : Int))
    off_dimcos_sin0 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range0[None, None, :] * $((p.stride_cosd : Int))
    off_dimcos_sin1 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range1[None, None, :] * $((p.stride_cosd : Int))
    k0 = tl.load(K + off_k0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    k1 = tl.load(K + off_k1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos0 = tl.load(Cos + off_dimcos_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin0 = tl.load(Sin + off_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos1 = tl.load(Cos + off_dimcos_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin1 = tl.load(Sin + off_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1
    tl.store(K + off_k0, out_k0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
    tl.store(K + off_k1, out_k1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
  }

/-- Existing candidate: only the integer head-block multiplication is commuted. -/
def indexCommuted (p : Layout) : ComputeKernel :=
  let Q := p.Q
  let K := p.K
  let Cos := p.Cos
  let Sin := p.Sin
  triton {
    tl.region Q = tl.float32, K = tl.float32, Cos = tl.float32, Sin = tl.float32
    cur_head_index = tl.program_id(0).to(tl.int64)
    cur_seq_index = tl.program_id(1).to(tl.int64)
    cur_head_range = $((p.BLOCK_HEAD : Int)) * cur_head_index + tl.arange(0, $(p.BLOCK_HEAD)).to(tl.int64)
    cur_seq_range = cur_seq_index * $((p.BLOCK_SEQ : Int)) + tl.arange(0, $(p.BLOCK_SEQ)).to(tl.int64)
    dim_range0 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64) * $((2 : Int))
    dim_range1 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64) * $((2 : Int)) + $((1 : Int))
    off_q0 = cur_seq_range[:, None, None] * $((p.stride_qbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_qh : Int)) +
      dim_range0[None, None, :] * $((p.stride_qd : Int))
    off_q1 = cur_seq_range[:, None, None] * $((p.stride_qbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_qh : Int)) +
      dim_range1[None, None, :] * $((p.stride_qd : Int))
    off_dimcos_sin0 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range0[None, None, :] * $((p.stride_cosd : Int))
    off_dimcos_sin1 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range1[None, None, :] * $((p.stride_cosd : Int))
    off_sin0 = cur_seq_range[:, None, None] * $((p.stride_sinbs : Int)) +
      dim_range0[None, None, :] * $((p.stride_sind : Int))
    off_sin1 = cur_seq_range[:, None, None] * $((p.stride_sinbs : Int)) +
      dim_range1[None, None, :] * $((p.stride_sind : Int))
    q0 = tl.load(Q + off_q0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    q1 = tl.load(Q + off_q1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos0 = tl.load(Cos + off_dimcos_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin0 = tl.load(Sin + off_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos1 = tl.load(Cos + off_dimcos_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin1 = tl.load(Sin + off_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    out0 = q0 * cos0 - q1 * sin0
    out1 = q0 * sin1 + q1 * cos1
    tl.store(Q + off_q0, out0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
    tl.store(Q + off_q1, out1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_Q : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
    off_k0 = cur_seq_range[:, None, None] * $((p.stride_kbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_kh : Int)) +
      dim_range0[None, None, :] * $((p.stride_kd : Int))
    off_k1 = cur_seq_range[:, None, None] * $((p.stride_kbs : Int)) +
      cur_head_range[None, :, None] * $((p.stride_kh : Int)) +
      dim_range1[None, None, :] * $((p.stride_kd : Int))
    off_dimcos_sin0 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range0[None, None, :] * $((p.stride_cosd : Int))
    off_dimcos_sin1 = cur_seq_range[:, None, None] * $((p.stride_cosbs : Int)) +
      dim_range1[None, None, :] * $((p.stride_cosd : Int))
    k0 = tl.load(K + off_k0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    k1 = tl.load(K + off_k1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos0 = tl.load(Cos + off_dimcos_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin0 = tl.load(Sin + off_sin0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    cos1 = tl.load(Cos + off_dimcos_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    sin1 = tl.load(Sin + off_sin1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))),
      other=0.0)
    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1
    tl.store(K + off_k0, out_k0,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
    tl.store(K + off_k1, out_k1,
      mask=(cur_seq_range[:, None, None] < $((p.max_total_len : Int))) &
      (cur_head_range[None, :, None] < $((p.HEAD_K : Int))) &
      (dim_range1[None, None, :] < $((p.ROTARY_DIM : Int))))
  }

end VeriTile.Bench.Repairs.RotaryEmb.Kernels.LegacyPastedSurface
