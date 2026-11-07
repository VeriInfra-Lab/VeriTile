import bench.tritonbench_g.rotary_transform.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Original Python bodies and existing candidates live in the shared Sources.Body entry.
Invocation adapters retain the existing public API and signed metadata widths.
The unused nheads/cache arguments are supplied from contiguous strides/sequence length. -/
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelSource
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def source32_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalTensorKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalScalarKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source32_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalTensorKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalScalarKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source64_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalTensorKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalScalarKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source64_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalTensorKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalScalarKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def shared32_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateTensorKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateScalarKernel Bool.false Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def shared32_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateTensorKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateScalarKernel Bool.false Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def shared64_32
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateTensorKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateScalarKernel Bool.true Bool.false OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def shared64_64
    (OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS : RegionName)
    (SEQLEN_OFFSETS_SCALAR seqlen : Int) (rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) : ComputeKernel :=
  if IS_SEQLEN_OFFSETS_TENSOR then
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateTensorKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
  else
    VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateScalarKernel Bool.true Bool.true OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS_SCALAR seqlen (stride_x_seqlen / stride_x_nheads) rotary_dim seqlen_ro (seqlen.toNat / 128) stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M IS_VARLEN INTERLEAVED CONJUGATE
def source (cumulative64 offsets64 : Bool) :=
  if cumulative64 then
    if offsets64 then source64_64 else source64_32
  else
    if offsets64 then source32_64 else source32_32

def shared (cumulative64 offsets64 : Bool) :=
  if cumulative64 then
    if offsets64 then shared64_64 else shared64_32
  else
    if offsets64 then shared32_64 else shared32_32

/-- The signed structural carrier is Int for either storage declaration.
This is the retained model equality, never permission to convert input bytes. -/
theorem signed_carrier (cumulative64 offsets64 : Bool) :
    source cumulative64 offsets64 = source64_64 := by
  cases cumulative64 <;> cases offsets64 <;> rfl

theorem shared_signed_carrier (cumulative64 offsets64 : Bool) :
    shared cumulative64 offsets64 = shared64_64 := by
  cases cumulative64 <;> cases offsets64 <;> rfl

end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelSource
