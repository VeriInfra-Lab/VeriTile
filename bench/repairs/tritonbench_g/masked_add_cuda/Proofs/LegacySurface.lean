import bench.repairs.tritonbench_g.masked_add_cuda.Kernels

namespace VeriTile.Bench.Repairs.MaskedAddCuda.LegacySurface
open VeriTile.Triton
open Lean
set_option hygiene false in
macro "maskedAddRepairSource%[" maskType:tritonDType "," variant:ident "]" : term => do
  let update ← if variant.getId == `original then
    `(tritonStmt| grad += p * $((alpha : ℝ)).to(tl.float32))
  else if variant.getId == `commuted then
    `(tritonStmt| grad = p * $((alpha : ℝ)).to(tl.float32) + grad)
  else Macro.throwError "expected original or commuted"
  `(triton {
    tl.region grad_ptr = tl.float32, p_ptr = tl.float32, p_mask_ptr = $maskType
    pid = tl.program_id(axis=0)
    block_start = pid * $(block)
    offsets = block_start + tl.arange(0, $(block))
    mask = offsets < $(n)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    $update:tritonStmt
    tl.store(grad_ptr + offsets, grad, mask=mask)
  })

/-- Original source with explicit FP32 regions and the two admitted mask widths. -/
def kernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  let grad_ptr := grad
  let p_ptr := data
  let p_mask_ptr := mask
  match storage with
  | .int32 => maskedAddRepairSource%[tl.int32, original]
  | .int64 => maskedAddRepairSource%[tl.int64, original]

/-- Actual retained candidate: move the product to the left of addition. -/
def commutedKernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  let grad_ptr := grad
  let p_ptr := data
  let p_mask_ptr := mask
  match storage with
  | .int32 => maskedAddRepairSource%[tl.int32, commuted]
  | .int64 => maskedAddRepairSource%[tl.int64, commuted]

end VeriTile.Bench.Repairs.MaskedAddCuda.LegacySurface
