import bench.tritonbench_g.rmsnorm_implementation.Sources.Mathematical_WholeSource

/-! Canonical selected FP32 source and the retained integer address-order
candidate. Both complete passes, epsilon, casts, reduction, sqrt and final
arithmetic order are unchanged. The selected native fusion graph is separate. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole.Parameters
abbrev Memory := VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole.Memory

def blockSize : Nat := 4096

/-- Complete selected FP32 source, with both full passes and the original
split square/add, reduction, square root, division and multiplication order. -/
def kernel (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, W], outputs=[O]
    tl.region X = tl.float32, W = tl.float32, O = tl.float32
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    offset_m = pid_batch * $(sxb) + pid_m * $(sxm)
    block_n_size = tl.arange(0, $(B))
    var = tl.zeros([$(B)], tl.float32)
    for block_n_strart_ptr in range(0, $(N), $(B)) {
      offset_n = block_n_strart_ptr + block_n_size
      x_ptr_mask = offset_n < $(N)
      x = tl.load(X + offset_m + offset_n * $(sxk), mask=x_ptr_mask, other=0.0)
      xf = (x).to(tl.float32)
      var += xf * xf
    }
    var = tl.sum(var, axis=0) / $(N)
    std = tl.sqrt(var + $(eps))
    for block_n_strart_ptr in range(0, $(N), $(B)) {
      offset_n = block_n_strart_ptr + block_n_size
      x_ptr_mask = offset_n < $(N)
      rms_w_offset = tl.load(W + offset_n * $(sw), mask=x_ptr_mask)
      x = tl.load(X + offset_m + offset_n * $(sxk), mask=x_ptr_mask, other=0.0)
      x_new = x / std
      out = x_new * rms_w_offset
      out_offset = pid_batch * $(sob) + pid_m * $(som) + offset_n * $(sok)
      tl.store(O + out_offset, out, mask=x_ptr_mask)
    }
  }

/-- Existing candidate changes only the batch-stride integer product order;
every numerical operation, cast, mask and omitted weight padding is retained. -/
def indexCommutedKernel (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, W], outputs=[O]
    tl.region X = tl.float32, W = tl.float32, O = tl.float32
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    offset_m = $(sxb) * pid_batch + pid_m * $(sxm)
    block_n_size = tl.arange(0, $(B))
    var = tl.zeros([$(B)], tl.float32)
    for block_n_strart_ptr in range(0, $(N), $(B)) {
      offset_n = block_n_strart_ptr + block_n_size
      x_ptr_mask = offset_n < $(N)
      x = tl.load(X + offset_m + offset_n * $(sxk), mask=x_ptr_mask, other=0.0)
      xf = (x).to(tl.float32)
      var += xf * xf
    }
    var = tl.sum(var, axis=0) / $(N)
    std = tl.sqrt(var + $(eps))
    for block_n_strart_ptr in range(0, $(N), $(B)) {
      offset_n = block_n_strart_ptr + block_n_size
      x_ptr_mask = offset_n < $(N)
      rms_w_offset = tl.load(W + offset_n * $(sw), mask=x_ptr_mask)
      x = tl.load(X + offset_m + offset_n * $(sxk), mask=x_ptr_mask, other=0.0)
      x_new = x / std
      out = x_new * rms_w_offset
      out_offset = pid_batch * $(sob) + pid_m * $(som) + offset_n * $(sok)
      tl.store(O + out_offset, out, mask=x_ptr_mask)
    }
  }

end VeriTile.Bench.Repairs.RmsnormImplementation
