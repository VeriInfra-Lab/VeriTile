import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.masked_add_cuda.Kernels

/-! Full native bodies for the selected repaired Python and its existing
addition-order candidate. The guarded launch uses BLOCK_SIZE=1024 and fusion.
Both mask widths are admitted. Runtime scalar ABI bindings are external to
the body; no extra source cast is inserted. Proofs/Graph interprets the same
observed fused lowering, without equating split rounding to FMA. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton

abbrev MaskStorage := VeriTile.Bench.TritonBenchG.MaskedAddCuda.MaskStorage

def blockSize : Nat := 1024

/-- FP32 payload and scalar ABI; signed int32 mask. Runtime integer
indices retain the existing mathematical projection. Only BLOCK_SIZE is constexpr. -/
def signature32 : CallSignature where
  pointers := [("grad_ptr",.fp32),("p_ptr",.fp32),("p_mask_ptr",.i32)]
  scalars := [("n_elements",.integer),("alpha",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["grad_ptr","p_ptr","p_mask_ptr"]
  writePorts := ["grad_ptr"]

def source32 (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call signature32 call_args [n_elements := n_elements, alpha := alpha] in
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

def commuted32 (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call signature32 call_args [n_elements := n_elements, alpha := alpha] in
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

/-- FP32 payload and scalar ABI; signed int64 mask. Runtime integer
indices retain the existing mathematical projection. Only BLOCK_SIZE is constexpr. -/
def signature64 : CallSignature where
  pointers := [("grad_ptr",.fp32),("p_ptr",.fp32),("p_mask_ptr",.i64)]
  scalars := [("n_elements",.integer),("alpha",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["grad_ptr","p_ptr","p_mask_ptr"]
  writePorts := ["grad_ptr"]

def source64 (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call signature64 call_args [n_elements := n_elements, alpha := alpha] in
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

def commuted64 (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call signature64 call_args [n_elements := n_elements, alpha := alpha] in
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

/-- Complete selected source; dispatch changes only the external mask ABI. -/
def kernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  match storage with
  | .int32 => source32 grad data mask n alpha block
  | .int64 => source64 grad data mask n alpha block

/-- Complete selected source; dispatch changes only the external mask ABI. -/
def commutedKernel (storage : MaskStorage) (grad data mask : RegionName)
    (n : Nat) (alpha : ℝ) (block : Nat := blockSize) : ComputeKernel :=
  match storage with
  | .int32 => commuted32 grad data mask n alpha block
  | .int64 => commuted64 grad data mask n alpha block

end VeriTile.Bench.Repairs.MaskedAddCuda
