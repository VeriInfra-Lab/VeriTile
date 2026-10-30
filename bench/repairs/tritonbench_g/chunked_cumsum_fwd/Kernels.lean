import bench.tritonbench_g.chunked_cumsum_fwd.Kernels
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.StatementAudit

/-! Unchanged complete selected Python body and the existing scale-first
candidate. Runtime clamps keep their FP32 registers; integer call values and
mixed interpreter storage channels are declared outside the raw quotations. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("dt_ptr",.fp32),("A_ptr",.fp32),("dt_bias_ptr",.fp32),("dt_out_ptr",.fp32),("dA_cumsum_ptr",.fp32)]
  scalars := [("batch",.integer),("seqlen",.integer),("nheads",.integer),("chunk_size",.integer),("dt_min",.fp32),("dt_max",.fp32),("stride_dt_batch",.integer),("stride_dt_seqlen",.integer),("stride_dt_head",.integer),("stride_A_head",.integer),("stride_dt_bias_head",.integer),("stride_dt_out_batch",.integer),("stride_dt_out_chunk",.integer),("stride_dt_out_head",.integer),("stride_dt_out_csize",.integer),("stride_dA_cs_batch",.integer),("stride_dA_cs_chunk",.integer),("stride_dA_cs_head",.integer),("stride_dA_cs_csize",.integer)]
  constexpr := ["DT_SOFTPLUS", "HAS_DT_BIAS", "BLOCK_SIZE_H", "BLOCK_SIZE_CHUNK"]
  typedFP32Pointers := ["dt_ptr", "A_ptr"]
  readPorts := ["dt_ptr", "A_ptr", "dt_bias_ptr"]
  writePorts := ["dt_out_ptr", "dA_cumsum_ptr"]

def source
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [batch := batch, seqlen := seqlen, nheads := nheads, chunk_size := chunk_size, stride_dt_batch := stride_dt_batch, stride_dt_seqlen := stride_dt_seqlen, stride_dt_head := stride_dt_head, stride_A_head := stride_A_head, stride_dt_bias_head := stride_dt_bias_head, stride_dt_out_batch := stride_dt_out_batch, stride_dt_out_chunk := stride_dt_out_chunk, stride_dt_out_head := stride_dt_out_head, stride_dt_out_csize := stride_dt_out_csize, stride_dA_cs_batch := stride_dA_cs_batch, stride_dA_cs_chunk := stride_dA_cs_chunk, stride_dA_cs_head := stride_dA_cs_head, stride_dA_cs_csize := stride_dA_cs_csize] in
  triton {
    pid_b = tl.program_id(axis=0).to(tl.uint64)
    pid_c = tl.program_id(axis=1).to(tl.uint64)
    pid_h = tl.program_id(axis=2).to(tl.uint64)
    # Keep index arithmetic unsigned and wide before products, including masked lanes.
    seqlen = tl.cast(seqlen, tl.uint64)
    nheads = tl.cast(nheads, tl.uint64)
    chunk_size = tl.cast(chunk_size, tl.uint64)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H).to(tl.uint64)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK).to(tl.uint64)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.log(1 + tl.exp(dt)), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
  }

def scaleFirst
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [batch := batch, seqlen := seqlen, nheads := nheads, chunk_size := chunk_size, stride_dt_batch := stride_dt_batch, stride_dt_seqlen := stride_dt_seqlen, stride_dt_head := stride_dt_head, stride_A_head := stride_A_head, stride_dt_bias_head := stride_dt_bias_head, stride_dt_out_batch := stride_dt_out_batch, stride_dt_out_chunk := stride_dt_out_chunk, stride_dt_out_head := stride_dt_out_head, stride_dt_out_csize := stride_dt_out_csize, stride_dA_cs_batch := stride_dA_cs_batch, stride_dA_cs_chunk := stride_dA_cs_chunk, stride_dA_cs_head := stride_dA_cs_head, stride_dA_cs_csize := stride_dA_cs_csize] in
  triton {
    pid_b = tl.program_id(axis=0).to(tl.uint64)
    pid_c = tl.program_id(axis=1).to(tl.uint64)
    pid_h = tl.program_id(axis=2).to(tl.uint64)
    # Keep index arithmetic unsigned and wide before products, including masked lanes.
    seqlen = tl.cast(seqlen, tl.uint64)
    nheads = tl.cast(nheads, tl.uint64)
    chunk_size = tl.cast(chunk_size, tl.uint64)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H).to(tl.uint64)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK).to(tl.uint64)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.log(1 + tl.exp(dt)), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = A[:, None] * dt
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
  }

end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel


namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd
/-- Historical literal-bound wrapper source retained for compatibility. -/
abbrev fp32Source := VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original .fp32 .fp32 .fp32
end VeriTile.Bench.Repairs.ChunkedCumsumFwd
