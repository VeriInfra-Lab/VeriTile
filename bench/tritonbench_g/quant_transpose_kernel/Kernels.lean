import bench.tritonbench_g.quant_transpose_kernel.Proofs.Sources.Mathematical_Kernels
import VeriTile.Triton.DSL.CallSignature

/-! The repair preserves the original scalar, grouped schedule,
strided load/store graph, two configurations and wrapper metadata. -/
namespace VeriTile.Bench.Repairs.QuantTransposeKernel

abbrev scalar := @QuantizationSource.scalar
abbrev schedule := QuantTransposeFinite.tileRow
end VeriTile.Bench.Repairs.QuantTransposeKernel

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource
open Lean VeriTile.Triton
abbrev Parameters := QuantTransposeFinite.Parameters

/-! Complete original Python body and the existing integer-address candidate.
The grouped schedule, rematerialization, masks, pointer updates and integer
conversion remain in the quotation. All runtime arguments are bound outside. -/
def callSignature : CallSignature where
  pointers := [("A",.fp32),("absmax_inv_ptr",.fp32),("B",.i8)]
  scalars := [("stride_am",.integer),("stride_an",.integer),("stride_bn",.integer),
    ("stride_bm",.integer),("M",.integer),("N",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","GROUP_M"]
  readPorts := ["A","absmax_inv_ptr"]
  writePorts := ["B"]

def quotedOriginalKernel (p : Parameters) : ComputeKernel :=
  let A : RegionName := ⟨p.x⟩
  let absmax_inv_ptr : RegionName := ⟨p.scale⟩
  let B : RegionName := ⟨p.output⟩
  let BLOCK_M : Nat := 128
  let BLOCK_N : Nat := 128
  let GROUP_M : Nat := 8
  let M := p.M
  let N := p.N
  let stride_am := p.sam
  let stride_an := p.san
  let stride_bm : Nat := 1
  let stride_bn := p.M
  with_triton_call callSignature call_args
    [stride_am := stride_am,stride_an := stride_an,stride_bn := stride_bn,
      stride_bm := stride_bm,M := M,N := N] in triton {
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

def quotedCommutedKernel (p : Parameters) : ComputeKernel :=
  let A : RegionName := ⟨p.x⟩
  let absmax_inv_ptr : RegionName := ⟨p.scale⟩
  let B : RegionName := ⟨p.output⟩
  let BLOCK_M : Nat := 128
  let BLOCK_N : Nat := 128
  let GROUP_M : Nat := 8
  let M := p.M
  let N := p.N
  let stride_am := p.sam
  let stride_an := p.san
  let stride_bm : Nat := 1
  let stride_bn := p.M
  with_triton_call callSignature call_args
    [stride_am := stride_am,stride_an := stride_an,stride_bn := stride_bn,
      stride_bm := stride_bm,M := M,N := N] in triton {
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

def kernel (version : Bool) : Parameters → ComputeKernel :=
  match version with
  | Bool.false => quotedOriginalKernel
  | Bool.true => quotedCommutedKernel

end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource
