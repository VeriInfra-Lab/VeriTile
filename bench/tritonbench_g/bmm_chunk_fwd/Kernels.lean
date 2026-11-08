import VeriTile.Triton.DSL.NestedReturns
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Sources.DTypes
namespace VeriTile.Bench.Repairs.BmmChunkFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
def signature (a_dtype b_dtype : Storage) (out_dtype : FloatDType) : CallSignature where
  pointers := [("a_ptr",match a_dtype with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),("b_ptr",match b_dtype with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),("seq_idx_ptr",.i64)]
  floatingPointers := [("out_ptr",out_dtype)]
  scalars := [("seqlen",.integer),("chunk_size",.integer),("K",.integer),("ngroups",.integer),("stride_a_batch",.integer),("stride_a_seqlen",.integer),("stride_a_head",.integer),("stride_ak",.integer),("stride_b_batch",.integer),("stride_b_seqlen",.integer),("stride_b_head",.integer),("stride_bk",.integer),("stride_out_batch",.integer),("stride_out_chunk",.integer),("stride_out_head",.integer),("stride_outm",.integer),("stride_outn",.integer),("stride_seq_idx_batch",.integer),("stride_seq_idx_seqlen",.integer)]
  constexpr := ["IS_CAUSAL","dot_dtype","HAS_SEQ_IDX","BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K"]
  readPorts := ["a_ptr","b_ptr","seq_idx_ptr"]
  writePorts := ["out_ptr"]

def originalKernel
    (a_dtype b_dtype : Storage) (out_dtype dot_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
with_triton_call (signature a_dtype b_dtype out_dtype) call_args [seqlen := seqlen, chunk_size := chunk_size, K := K, ngroups := ngroups, stride_a_batch := stride_a_batch, stride_a_seqlen := stride_a_seqlen, stride_a_head := stride_a_head, stride_ak := stride_ak, stride_b_batch := stride_b_batch, stride_b_seqlen := stride_b_seqlen, stride_b_head := stride_b_head, stride_bk := stride_bk, stride_out_batch := stride_out_batch, stride_out_chunk := stride_out_chunk, stride_out_head := stride_out_head, stride_outm := stride_outm, stride_outn := stride_outn, stride_seq_idx_batch := stride_seq_idx_batch, stride_seq_idx_seqlen := stride_seq_idx_seqlen] in triton {
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_b_batch = tl.cast(stride_b_batch, tl.int64)
    stride_b_seqlen = tl.cast(stride_b_seqlen, tl.int64)
    stride_b_head = tl.cast(stride_b_head, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_chunk = tl.cast(stride_out_chunk, tl.int64)
    stride_out_head = tl.cast(stride_out_head, tl.int64)
    stride_outm = tl.cast(stride_outm, tl.int64)
    stride_outn = tl.cast(stride_outn, tl.int64)
    stride_seq_idx_batch = tl.cast(stride_seq_idx_batch, tl.int64)
    stride_seq_idx_seqlen = tl.cast(stride_seq_idx_seqlen, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
  }

def indexCommutedKernel
    (a_dtype b_dtype : Storage) (out_dtype dot_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
with_triton_call (signature a_dtype b_dtype out_dtype) call_args [seqlen := seqlen, chunk_size := chunk_size, K := K, ngroups := ngroups, stride_a_batch := stride_a_batch, stride_a_seqlen := stride_a_seqlen, stride_a_head := stride_a_head, stride_ak := stride_ak, stride_b_batch := stride_b_batch, stride_b_seqlen := stride_b_seqlen, stride_b_head := stride_b_head, stride_bk := stride_bk, stride_out_batch := stride_out_batch, stride_out_chunk := stride_out_chunk, stride_out_head := stride_out_head, stride_outm := stride_outm, stride_outn := stride_outn, stride_seq_idx_batch := stride_seq_idx_batch, stride_seq_idx_seqlen := stride_seq_idx_seqlen] in triton {
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_b_batch = tl.cast(stride_b_batch, tl.int64)
    stride_b_seqlen = tl.cast(stride_b_seqlen, tl.int64)
    stride_b_head = tl.cast(stride_b_head, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_chunk = tl.cast(stride_out_chunk, tl.int64)
    stride_out_head = tl.cast(stride_out_head, tl.int64)
    stride_outm = tl.cast(stride_outm, tl.int64)
    stride_outn = tl.cast(stride_outn, tl.int64)
    stride_seq_idx_batch = tl.cast(stride_seq_idx_batch, tl.int64)
    stride_seq_idx_seqlen = tl.cast(stride_seq_idx_seqlen, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - ngroups * pid_c
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
  }

def dotFirstKernel
    (a_dtype b_dtype : Storage) (out_dtype dot_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
with_triton_call (signature a_dtype b_dtype out_dtype) call_args [seqlen := seqlen, chunk_size := chunk_size, K := K, ngroups := ngroups, stride_a_batch := stride_a_batch, stride_a_seqlen := stride_a_seqlen, stride_a_head := stride_a_head, stride_ak := stride_ak, stride_b_batch := stride_b_batch, stride_b_seqlen := stride_b_seqlen, stride_b_head := stride_b_head, stride_bk := stride_bk, stride_out_batch := stride_out_batch, stride_out_chunk := stride_out_chunk, stride_out_head := stride_out_head, stride_outm := stride_outm, stride_outn := stride_outn, stride_seq_idx_batch := stride_seq_idx_batch, stride_seq_idx_seqlen := stride_seq_idx_seqlen] in triton {
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_b_batch = tl.cast(stride_b_batch, tl.int64)
    stride_b_seqlen = tl.cast(stride_b_seqlen, tl.int64)
    stride_b_head = tl.cast(stride_b_head, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_chunk = tl.cast(stride_out_chunk, tl.int64)
    stride_out_head = tl.cast(stride_out_head, tl.int64)
    stride_outm = tl.cast(stride_outm, tl.int64)
    stride_outn = tl.cast(stride_outn, tl.int64)
    stride_seq_idx_batch = tl.cast(stride_seq_idx_batch, tl.int64)
    stride_seq_idx_seqlen = tl.cast(stride_seq_idx_seqlen, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc = tl.dot(a, b) + acc
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
  }

def clearSignature (out_dtype : FloatDType) : CallSignature where
  floatingPointers := [("Out",out_dtype)]
  scalars := [("chunk_size",.integer),("chunk_heads",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]
  readPorts := []
  writePorts := ["Out"]
def clearUpperKernel (out_dtype : FloatDType) (Out : RegionName)
    (chunk_size chunk_heads BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (clearSignature out_dtype) call_args [chunk_size := chunk_size, chunk_heads := chunk_heads] in triton {
    pid = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    chunk_head = tl.program_id(2).to(tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    chunk_heads = tl.cast(chunk_heads, tl.int64)
    tiles_n = tl.cdiv(chunk_size, BLOCK_N)
    row = pid // tiles_n * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    column = pid % tiles_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offsets = (batch * chunk_heads + chunk_head) * chunk_size * chunk_size + row[:, None] * chunk_size + column[None, :]
    tl.store(Out + offsets, 0.0, mask=(row[:, None] < chunk_size) & (column[None, :] < chunk_size) & (column[None, :] > row[:, None]))
  }
end
end VeriTile.Bench.Repairs.BmmChunkFwd
