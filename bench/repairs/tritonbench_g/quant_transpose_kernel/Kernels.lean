import bench.tritonbench_g.quant_transpose_kernel.Kernels

/-! The checked revision preserves the original scalar, grouped schedule,
strided load/store graph, two configurations and wrapper metadata. -/
namespace VeriTile.Bench.Repairs.QuantTransposeKernel
abbrev source := VeriTile.Bench.TritonBenchG.QuantTransposeKernel.quantize_global_transpose_real_surface
abbrev scalar := @QuantizationSource.scalar
abbrev schedule := QuantTransposeFinite.tileRow
end VeriTile.Bench.Repairs.QuantTransposeKernel

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource
open Lean VeriTile.Triton
abbrev Parameters := QuantTransposeFinite.Parameters

/-! The actual selected 128x128/G8 source, including its grouped one-dimensional
schedule, rematerialized coordinates, llrint64 and implicit int8 store. The
candidate only commutes an integer input-row address product. -/
def quotedOriginal (p : Parameters) : ComputeKernel :=
  let A : RegionName := ⟨p.x⟩
  let AbsmaxInv : RegionName := ⟨p.scale⟩
  let B : RegionName := ⟨p.output⟩
  triton {
    tl.region A = tl.float32, AbsmaxInv = tl.float32, B = tl.int8
    pid = tl.program_id(0)
    grid_m = ($((p.M : Nat)) + $(128) - $(1)) // $(128)
    grid_n = ($((p.N : Nat)) + $(128) - $(1)) // $(128)
    width = $(8) * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * $(8), $(8))
    pid_m = group_id * $(8) + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * $(128) + tl.arange(0, $(128))
    rn = pid_n * $(128) + tl.arange(0, $(128))
    A = A + (rm[:, None] * $((p.sam : Nat)) + rn[None, :] * $((p.san : Nat)))
    mask = (rm < $((p.M : Nat)))[:, None] & (rn < $((p.N : Nat)))[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(AbsmaxInv)
    rm = pid_m * $(128) + tl.arange(0, $(128))
    rn = pid_n * $(128) + tl.arange(0, $(128))
    B = B + (rm[:, None] * $(1) + rn[None, :] * $((p.M : Nat)))
    mask = (rm < $((p.M : Nat)))[:, None] & (rn < $((p.N : Nat)))[None, :]
    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
    tl.store(B, output, mask=mask)
  }

def quotedCommuted (p : Parameters) : ComputeKernel :=
  let A : RegionName := ⟨p.x⟩
  let AbsmaxInv : RegionName := ⟨p.scale⟩
  let B : RegionName := ⟨p.output⟩
  triton {
    tl.region A = tl.float32, AbsmaxInv = tl.float32, B = tl.int8
    pid = tl.program_id(0)
    grid_m = ($((p.M : Nat)) + $(128) - $(1)) // $(128)
    grid_n = ($((p.N : Nat)) + $(128) - $(1)) // $(128)
    width = $(8) * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * $(8), $(8))
    pid_m = group_id * $(8) + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * $(128) + tl.arange(0, $(128))
    rn = pid_n * $(128) + tl.arange(0, $(128))
    A = A + ($((p.sam : Nat)) * rm[:, None] + rn[None, :] * $((p.san : Nat)))
    mask = (rm < $((p.M : Nat)))[:, None] & (rn < $((p.N : Nat)))[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(AbsmaxInv)
    rm = pid_m * $(128) + tl.arange(0, $(128))
    rn = pid_n * $(128) + tl.arange(0, $(128))
    B = B + (rm[:, None] * $(1) + rn[None, :] * $((p.M : Nat)))
    mask = (rm < $((p.M : Nat)))[:, None] & (rn < $((p.N : Nat)))[None, :]
    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
    tl.store(B, output, mask=mask)
  }

def kernel (version : Bool) : Parameters → ComputeKernel :=
  match version with
  | Bool.false => quotedOriginal
  | Bool.true => quotedCommuted

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource
