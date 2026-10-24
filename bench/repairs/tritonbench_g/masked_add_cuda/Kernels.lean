import bench.tritonbench_g.masked_add_cuda.Kernels

/-! Canonical readable FP32 source and its existing addition-order variant.
The guarded Python launch fixes BLOCK_SIZE=1024 and enable_fp_fusion=True.
Fusion is interpreted by the selected native graph in Proofs/Graph; these
sources do not silently replace a fused launch with split-rounding arithmetic. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton

abbrev MaskStorage := VeriTile.Bench.TritonBenchG.MaskedAddCuda.MaskStorage

def blockSize : Nat := 1024

/-- Original source with explicit FP32 regions and the two admitted mask widths. -/
def kernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  let grad_ptr := grad
  let p_ptr := data
  let p_mask_ptr := mask
  match storage with
  | .int32 => triton {
    tl.region grad_ptr = tl.float32, p_ptr = tl.float32, p_mask_ptr = tl.int32
    pid = tl.program_id(axis=0)
    block_start = pid * $(block)
    offsets = block_start + tl.arange(0, $(block))
    mask = offsets < $(n)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * $((alpha : ℝ)).to(tl.float32)
    tl.store(grad_ptr + offsets, grad, mask=mask)
  }
  | .int64 => triton {
    tl.region grad_ptr = tl.float32, p_ptr = tl.float32, p_mask_ptr = tl.int64
    pid = tl.program_id(axis=0)
    block_start = pid * $(block)
    offsets = block_start + tl.arange(0, $(block))
    mask = offsets < $(n)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * $((alpha : ℝ)).to(tl.float32)
    tl.store(grad_ptr + offsets, grad, mask=mask)
  }

/-- Actual retained candidate: move the product to the left of addition. -/
def commutedKernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  let grad_ptr := grad
  let p_ptr := data
  let p_mask_ptr := mask
  match storage with
  | .int32 => triton {
    tl.region grad_ptr = tl.float32, p_ptr = tl.float32, p_mask_ptr = tl.int32
    pid = tl.program_id(axis=0)
    block_start = pid * $(block)
    offsets = block_start + tl.arange(0, $(block))
    mask = offsets < $(n)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad = p * $((alpha : ℝ)).to(tl.float32) + grad
    tl.store(grad_ptr + offsets, grad, mask=mask)
  }
  | .int64 => triton {
    tl.region grad_ptr = tl.float32, p_ptr = tl.float32, p_mask_ptr = tl.int64
    pid = tl.program_id(axis=0)
    block_start = pid * $(block)
    offsets = block_start + tl.arange(0, $(block))
    mask = offsets < $(n)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad = p * $((alpha : ℝ)).to(tl.float32) + grad
    tl.store(grad_ptr + offsets, grad, mask=mask)
  }

end VeriTile.Bench.Repairs.MaskedAddCuda
