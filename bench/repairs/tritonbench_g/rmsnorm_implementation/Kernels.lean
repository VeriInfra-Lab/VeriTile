import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.rmsnorm_implementation.WholeSource

/-! Canonical selected FP32 source and the retained integer address-order
candidate. Both complete passes, epsilon, casts, reduction, sqrt and final
arithmetic order are unchanged. The selected native fusion graph is separate. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole.Parameters
abbrev Memory := VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole.Memory

def blockSize : Nat := 4096

/-- Native Python invocation profile. Strides remain runtime arguments;
N_SIZE, eps and BLOCK_N_SIZE retain their original constexpr roles. -/
def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32), ("rms_w_ptr",.fp32), ("out_ptr",.fp32)]
  scalars := [("stride_x_batch",.integer), ("stride_x_m",.integer), ("stride_x_k",.integer), ("stride_rms_w",.integer), ("stride_out_batch",.integer), ("stride_out_m",.integer), ("stride_out_k",.integer)]
  constexpr := ["N_SIZE", "eps", "BLOCK_N_SIZE"]
  readPorts := ["x_ptr", "rms_w_ptr"]
  writePorts := ["out_ptr"]

/-- Complete selected FP32 source, with both full passes and the original
split square/add, reduction, square root, division and multiplication order. -/
def kernel (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let x_ptr : RegionName := X
  let rms_w_ptr : RegionName := W
  let out_ptr : RegionName := O
  let N_SIZE : Nat := N
  let BLOCK_N_SIZE : Nat := B
  with_triton_call callSignature call_args [stride_x_batch := sxb, stride_x_m := sxm, stride_x_k := sxk, stride_rms_w := sw, stride_out_batch := sob, stride_out_m := som, stride_out_k := sok] in
  triton {
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # parallel at m dimension
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    # parallel between blocks
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)  # careful stride_x_k
        xf = x.to(tl.float32)
        var += xf*xf
    var = tl.sum(var, axis=0) / N_SIZE  # reduce between wrap
    std = tl.sqrt(var + eps)

    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE

        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)

        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }

/-- Existing candidate: only the integer batch-stride product is commuted. -/
def indexCommutedKernel (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let x_ptr : RegionName := X
  let rms_w_ptr : RegionName := W
  let out_ptr : RegionName := O
  let N_SIZE : Nat := N
  let BLOCK_N_SIZE : Nat := B
  with_triton_call callSignature call_args [stride_x_batch := sxb, stride_x_m := sxm, stride_x_k := sxk, stride_rms_w := sw, stride_out_batch := sob, stride_out_m := som, stride_out_k := sok] in
  triton {
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # parallel at m dimension
    offset_m = stride_x_batch * pid_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    # parallel between blocks
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)  # careful stride_x_k
        xf = x.to(tl.float32)
        var += xf*xf
    var = tl.sum(var, axis=0) / N_SIZE  # reduce between wrap
    std = tl.sqrt(var + eps)

    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE

        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)

        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }

end VeriTile.Bench.Repairs.RmsnormImplementation
