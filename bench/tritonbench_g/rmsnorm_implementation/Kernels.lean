import VeriTile.Triton.DSL.PythonCallSyntax
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Sources.Parameters

/-! Complete unchanged Python bodies for every retained storage profile.
Strides are runtime integers; N_SIZE, eps and BLOCK_N_SIZE are constexpr.
The existing candidate commutes only the first batch-stride product. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation (Storage Source)
abbrev Parameters := VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole.Parameters
abbrev Memory := VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole.Memory

def blockSize : Nat := 4096

def callSignature (input weight output : Storage) : CallSignature where
  pointers := [("x_ptr",match input with | .fp32 => .fp32 | .fp64 => .fp64 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("rms_w_ptr",match weight with | .fp32 => .fp32 | .fp64 => .fp64 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("out_ptr",match output with | .fp32 => .fp32 | .fp64 => .fp64 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  scalars := [("stride_x_batch",.integer),("stride_x_m",.integer),("stride_x_k",.integer),
    ("stride_rms_w",.integer),("stride_out_batch",.integer),("stride_out_m",.integer),("stride_out_k",.integer)]
  constexpr := ["N_SIZE","eps","BLOCK_N_SIZE"]
  readPorts := ["x_ptr","rms_w_ptr"]
  writePorts := ["out_ptr"]

set_option maxHeartbeats 1000000 in
def originalForKernel (input weight output : Storage) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let x_ptr := X
  let rms_w_ptr := W
  let out_ptr := O
  let N_SIZE := N
  let BLOCK_N_SIZE := B
  with_triton_call (callSignature input weight output) call_args
    [stride_x_batch := sxb,stride_x_m := sxm,stride_x_k := sxk,stride_rms_w := sw,
      stride_out_batch := sob,stride_out_m := som,stride_out_k := sok] in triton {
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

set_option maxHeartbeats 1000000 in
def commutedForKernel (input weight output : Storage) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let x_ptr := X
  let rms_w_ptr := W
  let out_ptr := O
  let N_SIZE := N
  let BLOCK_N_SIZE := B
  with_triton_call (callSignature input weight output) call_args
    [stride_x_batch := sxb,stride_x_m := sxm,stride_x_k := sxk,stride_rms_w := sw,
      stride_out_batch := sob,stride_out_m := som,stride_out_k := sok] in triton {
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

def family (input weight output : Storage) (candidate : Bool) : Source :=
  if candidate then commutedForKernel input weight output else originalForKernel input weight output

def kernel (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  originalForKernel .fp32 .fp32 .fp32 X W O sxb sxm sxk sw sob som sok N B eps

def indexCommutedKernel (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  commutedForKernel .fp32 .fp32 .fp32 X W O sxb sxm sxk sw sob som sok N B eps
end VeriTile.Bench.Repairs.RmsnormImplementation
