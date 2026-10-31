import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! The shared selected FP32 input source; output and residual dtype parameters
retain their existing ranges. Whole-function identity was checked before
retiring the independent selected specialization. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd.Kernels
open Lean VeriTile.Triton
def quotedOriginal :=
  fun (dtype out_dtype : FloatDType) (a_ptr dout_ptr db_ptr res_ptr : RegionName)
      (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n
        stride_db_batch stride_db_seqlen stride_db_head stride_db_k
        stride_res_batch stride_res_seqlen stride_res_head stride_res_k
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) =>
  triton {
    tl.ports inputs=[a_ptr, dout_ptr, res_ptr], outputs=[db_ptr]
    tl.region a_ptr = tl.float32, dout_ptr = tl.float32, res_ptr = DTYPE, db_ptr = OUT_DTYPE
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // $(ngroups)
    pid_h = pid_ch - pid_c * $(ngroups)
    num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
    dout_ptr += pid_b * $(stride_dout_batch) + pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
    offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
    offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
    dout_ptrs = dout_ptr + (offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m))
    a_ptrs = a_ptr + (offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak))
    chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
    acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    for cs in range($(0), tl.cdiv(chunk_size_limit, $(BLOCK_SIZE_CS)), $(1)) {
      dout = (tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
        (offs_cs[None, :] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)), other=0.0)).to(tl.float32)
      a = (tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)) &
        (offs_n[None, :] < $(K)), other=0.0)).to(tl.float32)
      acc += tl.dot(dout, a)
      dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
      a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
    }
    offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
    offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    if HAS_RESIDUAL {
      res_ptr += pid_b * $(stride_res_batch) + pid_c * $(chunk_size) * $(stride_res_seqlen) + pid_h * $(stride_res_head)
      res_ptrs = res_ptr + (offs_m[:, None] * $(stride_res_seqlen) + offs_n[None, :] * $(stride_res_k))
      res = (tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
        (offs_n[None, :] < $(K)))).to(tl.float32)
      acc += res
    }
    db = (acc).to(OUT_DTYPE)
    db_ptr += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
    db_ptrs = db_ptr + (offs_m[:, None] * $(stride_db_seqlen) + offs_n[None, :] * $(stride_db_k))
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < $(K)))
  }
end VeriTile.Bench.Repairs.BmmChunkBwd.Kernels
