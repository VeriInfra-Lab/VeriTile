import bench.tritonbench_g.rotary_transform.Kernels
import bench.tritonbench_g.rotary_transform_ops.Proofs.Support.Model
import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Original Python bodies and existing candidates live in the shared Sources.Body entry.
Invocation adapters retain the existing public API and signed metadata widths.
The unused nheads/cache arguments are supplied from contiguous strides/sequence length. -/

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelSource
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

/-- Invocation adapters for all four signed metadata storage combinations.
Each delegates to its complete copied Python body in the shared source entry. -/
def source32_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source32_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source64_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source64_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source (cumulative64 offsets64 : Bool) :=
  if cumulative64 then
    if offsets64 then source64_64 else source64_32
  else
    if offsets64 then source32_64 else source32_32
end VeriTile.Bench.Repairs.RotaryTransformOps.KernelSource


namespace VeriTile.Bench.Repairs.RotaryTransformOps.XorSource
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

/-- Complete ordinary source quotations for the four existing signed metadata
storage combinations. The sole arithmetic change is exact adjacent-partner indexing with xor one. -/
def source32_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source32_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source64_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source64_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source (cumulative64 offsets64 : Bool) :=
  if cumulative64 then
    if offsets64 then source64_64 else source64_32
  else
    if offsets64 then source32_64 else source32_32
end VeriTile.Bench.Repairs.RotaryTransformOps.XorSource

namespace VeriTile.Bench.Repairs.RotaryTransformOps.Kernels
/-- Complete original program constructor, including both metadata widths. -/
abbrev original := KernelSource.source
/-- Complete XOR-indexed candidate with the same metadata choices. -/
abbrev xorIndexed := XorSource.source
end VeriTile.Bench.Repairs.RotaryTransformOps.Kernels
