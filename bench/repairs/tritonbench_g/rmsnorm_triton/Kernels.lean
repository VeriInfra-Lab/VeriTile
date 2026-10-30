import bench.tritonbench_g.rmsnorm_triton.WholeSource

/-! Current checked source: FP32 input, weight, accumulation and output. Both
1024-lane loops, the integer pow call, explicit casts and masked weight load
are retained. The candidate commutes one integer address product only. -/
namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.Parameters
abbrev Memory := VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.Memory

def blockSize : Nat := 1024

def kernel (x_ptr rms_w_ptr output_ptr : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[x_ptr, rms_w_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, rms_w_ptr = tl.float32, output_ptr = tl.float32
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_batch * $(sxb) + pid_m * $(sxm)
    block_N = tl.arange(0, $(B))
    var = tl.zeros([$(B)], tl.float32)
    for block_n_start_idx in range(0, $(N), $(B)) {
      offs_n = block_n_start_idx + block_N
      x_ptr_mask = offs_n < $(N)
      x = tl.load(x_ptr + offs_m + offs_n * $(sxk), mask=x_ptr_mask, other=0.0)
      var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
    }
    var = tl.sum(var, axis=0) / $(N)
    rstd = tl.math.rsqrt(var + $(eps))
    for block_n_start_idx in range(0, $(N), $(B)) {
      offs_n = block_n_start_idx + block_N
      x_ptr_mask = offs_n < $(N)
      rms_w = tl.load(rms_w_ptr + offs_n * $(sw), mask=x_ptr_mask)
      x = tl.load(x_ptr + offs_m + offs_n * $(sxk), mask=x_ptr_mask, other=0.0).to(tl.float32)
      x_hat = x * rstd
      out = x_hat * rms_w
      out_off = pid_batch * $(sob) + pid_m * $(som) + offs_n * $(sok)
      tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
    }
  }

/-- Only the natural address product changes; every FP statement is retained. -/
def indexCommutedKernel (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, W], outputs=[O]
    tl.region X = tl.float32, W = tl.float32, O = tl.float32
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = $(sxb) * pid_batch + pid_m * $(sxm)
    block_N = tl.arange(0, $(B))
    var = tl.zeros([$(B)], tl.float32)
    for block_n_start_idx in range(0, $(N), $(B)) {
      offs_n = block_n_start_idx + block_N
      x_ptr_mask = offs_n < $(N)
      x = tl.load(X + offs_m + offs_n * $(sxk), mask=x_ptr_mask, other=0.0)
      var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
    }
    var = tl.sum(var, axis=0) / $(N)
    rstd = tl.math.rsqrt(var + $(eps))
    for block_n_start_idx in range(0, $(N), $(B)) {
      offs_n = block_n_start_idx + block_N
      x_ptr_mask = offs_n < $(N)
      rms_w = tl.load(W + offs_n * $(sw), mask=x_ptr_mask)
      x = tl.load(X + offs_m + offs_n * $(sxk), mask=x_ptr_mask, other=0.0).to(tl.float32)
      x_hat = x * rstd
      out = x_hat * rms_w
      out_off = pid_batch * $(sob) + pid_m * $(som) + offs_n * $(sok)
      tl.store(O + out_off, out, mask=x_ptr_mask)
    }
  }

end VeriTile.Bench.Repairs.RmsnormTriton
