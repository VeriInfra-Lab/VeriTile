import bench.tritonbench_g.bmm_chunk_fwd.Source

namespace VeriTile.Bench.Repairs.BmmChunkFwd
open Lean VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
namespace Kernels

/-- Complete admitted FP32 main source. The two-argument dot and following
outer accumulator addition retain the original source graph. -/
def mainSource
    (out_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
    triton {
      tl.ports inputs=[a_ptr, b_ptr, Region.cast seq_idx_ptr], outputs=[out_ptr]
      tl.region a_ptr = tl.float32, b_ptr = tl.float32, out_ptr = OUT_DTYPE
      pid_b = tl.program_id(axis=1)
      pid_ch = tl.program_id(axis=2)
      pid_c = pid_ch // $(ngroups)
      pid_h = pid_ch - pid_c * $(ngroups)
      num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
      pid_m = tl.program_id(axis=0) // num_pid_n
      pid_n = tl.program_id(axis=0) % num_pid_n
      active_block = true
      if IS_CAUSAL {
        active_block = pid_n * $(BLOCK_SIZE_N) < (pid_m + $(1)) * $(BLOCK_SIZE_M)
      }
      if active_block {
        a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
        b_ptr += pid_b * $(stride_b_batch) + pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
        if HAS_SEQ_IDX {
          seq_idx_ptr += pid_b * $(stride_seq_idx_batch) + pid_c * $(chunk_size) * $(stride_seq_idx_seqlen)
        }
        offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
        offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
        offs_k = tl.arange(0, $(BLOCK_SIZE_K))
        a_ptrs = a_ptr + (offs_m[:, None] * $(stride_a_seqlen) + offs_k[None, :] * $(stride_ak))
        b_ptrs = b_ptr + (offs_k[:, None] * $(stride_bk) + offs_n[None, :] * $(stride_b_seqlen))
        chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
        acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
        for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
          a = (tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0)).to(tl.float32)
          b = (tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
            (offs_n[None, :] < chunk_size_limit), other=0.0)).to(tl.float32)
          acc += tl.dot(a, b)
          a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
          b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
        }
        offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
        offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
        if HAS_SEQ_IDX {
          chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
          seq_idx_m = tl.load(seq_idx_ptr + offs_m * $(stride_seq_idx_seqlen), mask=offs_m < chunk_size_limit, other=-1)
          seq_idx_n = tl.load(seq_idx_ptr + offs_n * $(stride_seq_idx_seqlen), mask=offs_n < chunk_size_limit, other=-2)
          acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
        }
        out = (acc).to(OUT_DTYPE)
        out_ptr += pid_b * $(stride_out_batch) + pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
        out_ptrs = out_ptr + ($(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn))
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) & (offs_n[None, :] < $(chunk_size)))
      }
    }

/-- Existing integer head-index candidate; all floating operations, causal
control, sequence filtering, output conversion and masks are unchanged. -/
def mainIndexCommuted
    (out_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
    triton {
      tl.ports inputs=[a_ptr, b_ptr, Region.cast seq_idx_ptr], outputs=[out_ptr]
      tl.region a_ptr = tl.float32, b_ptr = tl.float32, out_ptr = OUT_DTYPE
      pid_b = tl.program_id(axis=1)
      pid_ch = tl.program_id(axis=2)
      pid_c = pid_ch // $(ngroups)
      pid_h = pid_ch - $(ngroups) * pid_c
      num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
      pid_m = tl.program_id(axis=0) // num_pid_n
      pid_n = tl.program_id(axis=0) % num_pid_n
      active_block = true
      if IS_CAUSAL {
        active_block = pid_n * $(BLOCK_SIZE_N) < (pid_m + $(1)) * $(BLOCK_SIZE_M)
      }
      if active_block {
        a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
        b_ptr += pid_b * $(stride_b_batch) + pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
        if HAS_SEQ_IDX {
          seq_idx_ptr += pid_b * $(stride_seq_idx_batch) + pid_c * $(chunk_size) * $(stride_seq_idx_seqlen)
        }
        offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
        offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
        offs_k = tl.arange(0, $(BLOCK_SIZE_K))
        a_ptrs = a_ptr + (offs_m[:, None] * $(stride_a_seqlen) + offs_k[None, :] * $(stride_ak))
        b_ptrs = b_ptr + (offs_k[:, None] * $(stride_bk) + offs_n[None, :] * $(stride_b_seqlen))
        chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
        acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
        for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
          a = (tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0)).to(tl.float32)
          b = (tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
            (offs_n[None, :] < chunk_size_limit), other=0.0)).to(tl.float32)
          acc += tl.dot(a, b)
          a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
          b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
        }
        offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
        offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
        if HAS_SEQ_IDX {
          chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
          seq_idx_m = tl.load(seq_idx_ptr + offs_m * $(stride_seq_idx_seqlen), mask=offs_m < chunk_size_limit, other=-1)
          seq_idx_n = tl.load(seq_idx_ptr + offs_n * $(stride_seq_idx_seqlen), mask=offs_n < chunk_size_limit, other=-2)
          acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
        }
        out = (acc).to(OUT_DTYPE)
        out_ptr += pid_b * $(stride_out_batch) + pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
        out_ptrs = out_ptr + ($(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn))
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) & (offs_n[None, :] < $(chunk_size)))
      }
    }

def bindMain (version : Bool) (h : Host.Call) (c : Host.Config) :=
  (if version then mainIndexCommuted else mainSource)
    (Host.outputDType h.aStorage h.requestedOutput) h.A h.B h.Out h.Seq
    h.shape.seqlen h.shape.chunk h.shape.K h.shape.groups
    (Host.prepared h.shape h.aStrides).batch (Host.prepared h.shape h.aStrides).row
    (Host.prepared h.shape h.aStrides).head (Host.prepared h.shape h.aStrides).column
    (Host.prepared h.shape h.bStrides).batch (Host.prepared h.shape h.bStrides).row
    (Host.prepared h.shape h.bStrides).head (Host.prepared h.shape h.bStrides).column
    ((h.shape.chunks * h.shape.groups) * h.shape.chunk * h.shape.chunk)
    (h.shape.groups * h.shape.chunk * h.shape.chunk)
    (if h.shape.grouped then h.shape.chunk * h.shape.chunk else 0) h.shape.chunk 1
    (if h.hasSeq then h.seqBatch else 0) (if h.hasSeq then h.seqRow else 0)
    c.M c.N c.K h.causal h.hasSeq


def fp32Source := bindMain Bool.false
def indexCommuted := bindMain Bool.true

/-- Existing general typed source retained for the accepted proof library. -/
abbrev source (h : Host.Call) (c : Host.Config) := h.kernel c Bool.false
abbrev configurations := Host.configurations

theorem fp32Source_eq (h : Host.Call) (c : Host.Config)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    fp32Source h c=source h c := by
  simp only [fp32Source,bindMain,source,Host.Call.kernel,ha,hb,TypedSource.specialize,mainSource,
    Bool.false_eq_true,↓reduceIte] <;> rfl

/-- The actual final 32 by 64 cleanup source. Int64 casts have the same
mathematical erasure as main; NativeCleanup proves finite source arithmetic. -/
def clearUpper (Out : RegionName) (chunk_size chunk_heads : Nat) : ComputeKernel :=
  let out_dtype : FloatDType := .fp32
  triton {
    tl.region Out = OUT_DTYPE
    pid = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    chunk_head = tl.program_id(axis=2)
    tiles_n = tl.cdiv($(chunk_size), $(64))
    row = pid // tiles_n * $(32) + tl.arange(0, $(32))
    column = pid % tiles_n * $(64) + tl.arange(0, $(64))
    offsets = (batch * $(chunk_heads) + chunk_head) * $(chunk_size) * $(chunk_size) +
      row[:, None] * $(chunk_size) + column[None, :]
    tl.store(Out + offsets, (0.0).to(OUT_DTYPE),
      mask=(row[:, None] < $(chunk_size)) & (column[None, :] < $(chunk_size)) &
        (column[None, :] > row[:, None]))
  }

#axiomsClean fp32Source_eq
end Kernels
end VeriTile.Bench.Repairs.BmmChunkFwd
