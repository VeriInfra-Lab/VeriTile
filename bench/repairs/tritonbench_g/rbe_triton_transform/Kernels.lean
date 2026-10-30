import VeriTile.Triton

namespace VeriTile.Bench.Repairs.RbeTritonTransform
open VeriTile.Triton
noncomputable section

structure Request where
  batches : Nat
  rows : Nat
  columns : Nat
  position : Int

def Request.size (r : Request) := r.batches*r.rows*r.columns
def Request.offset (r : Request) (b m c : Nat) := (b*r.rows+m)*r.columns+c

namespace Source
structure Parameters where
  (input output : RegionName)
  (rows columns inputBatchStride inputRowStride inputColumnStride
    outputBatchStride outputRowStride outputColumnStride : Nat)
  startToken : Int
  theta : ℝ
  (frequencyDimension blockRows blockColumns : Nat)

def typed (dtype : FloatDType)
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      : Nat)
    (start_token_position : Int)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  tl.region x_ptr = DTYPE, out_ptr = DTYPE
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(K), $(BLOCK_SIZE_K))
  pid_n = pid % tl.cdiv($(K), $(BLOCK_SIZE_K))

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * $(stride_x_batch) + $(stride_x_m) * offs_m[:, None] +
    $(stride_x_n) * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  imag = tl.load(x_ptrs + $(stride_x_n), mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = $((start_token_position : Int)) + pid_m * $(BLOCK_SIZE_M)
  freqs = offs_n % $(DIM)
  freqs_f = tl.toReal(freqs).to(tl.float32) / $(DIM)
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $(BLOCK_SIZE_M)) + start_block
  tks_f = tl.toReal(tks).to(tl.float32)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real.to(tl.float32) * cos - imag.to(tl.float32) * sin
  out_imag = real.to(tl.float32) * sin + imag.to(tl.float32) * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * $(stride_out_batch) + $(stride_out_m) * offs_m[:, None] +
    $(stride_out_n) * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  tl.store(out_ptrs, out_real.to(DTYPE), mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  tl.store(out_ptrs + $(stride_out_n), out_imag.to(DTYPE), mask=out_imag_mask)
}

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

end Source

abbrev parameters (r : Request) (input output : RegionName) : Source.Parameters :=
  ⟨input,output,r.rows,r.columns,r.rows*r.columns,r.columns,1,
    r.rows*r.columns,r.columns,1,r.position,10000,128,2,1024⟩

def kernel (r : Request) (input output : RegionName) := Source.source (parameters r input output)

end
end VeriTile.Bench.Repairs.RbeTritonTransform
