import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.kldiv_compute.Proofs.Support.Parameters

/-! All retained floating storage profiles share these two unchanged raw
Python bodies. Invocation metadata supplies dtypes; the existing support
classifier continues to distinguish unsupported native fp8-input pairs. -/
namespace VeriTile.Bench.Repairs.KldivCompute.KernelDSL
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource

def callSignature (xType yType outType : Storage) : CallSignature where
  pointers := [("x_ptr",match xType with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("y_ptr",match yType with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("output_ptr",match outType with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  scalars := [("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE"]

-- Elaborate the finite 6×6×6 caller dtype choices for this one body.
set_option maxHeartbeats 1000000 in
def originalForKernel (xType yType outType : Storage) (x_ptr y_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (callSignature xType yType outType) call_args [n_elements := N] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

set_option maxHeartbeats 1000000 in
def commutedForKernel (xType yType outType : Storage) (x_ptr y_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (callSignature xType yType outType) call_args [n_elements := N] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def family (xType yType outType : Storage) (version : Bool) :
    RegionName → RegionName → RegionName → Nat → Nat → ComputeKernel :=
  if version then commutedForKernel xType yType outType else originalForKernel xType yType outType

def kernel (p : Parameters) : ComputeKernel :=
  originalForKernel .fp32 .fp32 .fp32 p.x p.y p.output p.n p.block

def commutedKernel (p : Parameters) : ComputeKernel :=
  commutedForKernel .fp32 .fp32 .fp32 p.x p.y p.output p.n p.block
end VeriTile.Bench.Repairs.KldivCompute.KernelDSL
