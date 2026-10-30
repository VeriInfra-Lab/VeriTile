import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.rmsnorm_triton.WholeSource

/-! Current checked source: FP32 input, weight, accumulation and output. Both
1024-lane loops, the integer pow call, explicit casts and masked weight load
are retained. The candidate commutes one integer address product only. -/
namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.Parameters
abbrev Memory := VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.Memory

def blockSize : Nat := 1024

/-- Native Python invocation profile. Strides remain runtime arguments;
N_SIZE, eps and BLOCK_N_SIZE retain their original constexpr roles. -/
def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32), ("rms_w_ptr",.fp32), ("output_ptr",.fp32)]
  scalars := [("stride_x_batch",.integer), ("stride_x_m",.integer), ("stride_x_k",.integer), ("stride_rms_w",.integer), ("stride_out_batch",.integer), ("stride_out_m",.integer), ("stride_out_k",.integer)]
  constexpr := ["N_SIZE", "eps", "BLOCK_N_SIZE"]
  readPorts := ["x_ptr", "rms_w_ptr"]
  writePorts := ["output_ptr"]

def kernel (x_ptr rms_w_ptr output_ptr : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let N_SIZE : Nat := N
  let BLOCK_N_SIZE : Nat := B
  with_triton_call callSignature call_args [stride_x_batch := sxb, stride_x_m := sxm, stride_x_k := sxk, stride_rms_w := sw, stride_out_batch := sob, stride_out_m := som, stride_out_k := sok] in
  triton {
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }

/-- Existing candidate: only the integer batch-stride product is commuted. -/
def indexCommutedKernel (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let x_ptr : RegionName := X
  let rms_w_ptr : RegionName := W
  let output_ptr : RegionName := O
  let N_SIZE : Nat := N
  let BLOCK_N_SIZE : Nat := B
  with_triton_call callSignature call_args [stride_x_batch := sxb, stride_x_m := sxm, stride_x_k := sxk, stride_rms_w := sw, stride_out_batch := sob, stride_out_m := som, stride_out_k := sok] in
  triton {
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = stride_x_batch * pid_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }

end VeriTile.Bench.Repairs.RmsnormTriton
