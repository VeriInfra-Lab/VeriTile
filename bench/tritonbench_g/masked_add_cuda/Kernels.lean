import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.masked_add_cuda.Proofs.Sources.Parameters

/-! Complete unchanged Python bodies. The two mask storage choices are
caller metadata; the existing fused launch model remains in Proofs/Graph. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton
abbrev MaskStorage := VeriTile.Bench.TritonBenchG.MaskedAddCuda.MaskStorage

def blockSize : Nat := 1024

def sourceSignature (storage : MaskStorage) : CallSignature where
  pointers := [("grad_ptr",.fp32),("p_ptr",.fp32),
    ("p_mask_ptr",match storage with | .int32 => .i32 | .int64 => .i64)]
  scalars := [("n_elements",.integer),("alpha",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["p_mask_ptr","p_ptr","grad_ptr"]
  writePorts := ["grad_ptr"]

def sourceKernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  let grad_ptr := grad
  let p_ptr := data
  let p_mask_ptr := mask
  let BLOCK_SIZE := block
  with_triton_call (sourceSignature storage) call_args [n_elements := n,alpha := alpha] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * alpha
    tl.store(grad_ptr + offsets, grad, mask=mask)
  }

def commutedKernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  let grad_ptr := grad
  let p_ptr := data
  let p_mask_ptr := mask
  let BLOCK_SIZE := block
  with_triton_call (sourceSignature storage) call_args [n_elements := n,alpha := alpha] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad = p * alpha + grad
    tl.store(grad_ptr + offsets, grad, mask=mask)
  }

end VeriTile.Bench.Repairs.MaskedAddCuda
