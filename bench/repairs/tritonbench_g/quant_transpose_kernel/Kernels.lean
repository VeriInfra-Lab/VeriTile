import bench.tritonbench_g.quant_transpose_kernel.Kernels
import VeriTile.Triton.DSL.CallSignature

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
def sourceSignature : CallSignature where
  pointers := [("A",.fp32),("absmax_inv_ptr",.fp32),("B",.i8)]
  scalars := [("stride_am",.i32),("stride_an",.i32),("stride_bn",.i32),
    ("stride_bm",.i32),("M",.i32),("N",.i32)]
  constexpr := ["BLOCK_M","BLOCK_N","GROUP_M"]
  readPorts := ["A","absmax_inv_ptr"]
  writePorts := ["B"]

def sourceKernel (A absmax_inv_ptr B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) : ComputeKernel :=
  let stride_am_value : Nat := stride_am
  let stride_an_value : Nat := stride_an
  let stride_bn_value : Nat := stride_bn
  let stride_bm_value : Nat := stride_bm
  let M_value : Nat := M
  let N_value : Nat := N
  with_triton_call sourceSignature call_args
    [stride_am := stride_am_value, stride_an := stride_an_value, stride_bn := stride_bn_value, stride_bm := stride_bm_value, M := M_value, N := N_value] in
  triton {
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)

    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))

    tl.store(B, output, mask=mask)
  }

def commutedSourceKernel (A absmax_inv_ptr B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) : ComputeKernel :=
  let stride_am_value : Nat := stride_am
  let stride_an_value : Nat := stride_an
  let stride_bn_value : Nat := stride_bn
  let stride_bm_value : Nat := stride_bm
  let M_value : Nat := M
  let N_value : Nat := N
  with_triton_call sourceSignature call_args
    [stride_am := stride_am_value, stride_an := stride_an_value, stride_bn := stride_bn_value, stride_bm := stride_bm_value, M := M_value, N := N_value] in
  triton {
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (stride_am * rm[:, None] + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)

    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))

    tl.store(B, output, mask=mask)
  }

/-- Full selected 128x128/G8 call, retaining arbitrary admitted source strides. -/
def quotedOriginal (p : Parameters) : ComputeKernel :=
  sourceKernel ⟨p.x⟩ ⟨p.scale⟩ ⟨p.output⟩ p.sam p.san p.M 1 p.M p.N 128 128 8

def quotedCommuted (p : Parameters) : ComputeKernel :=
  commutedSourceKernel ⟨p.x⟩ ⟨p.scale⟩ ⟨p.output⟩ p.sam p.san p.M 1 p.M p.N 128 128 8

def kernel (version : Bool) : Parameters → ComputeKernel :=
  match version with
  | Bool.false => quotedOriginal
  | Bool.true => quotedCommuted

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource
