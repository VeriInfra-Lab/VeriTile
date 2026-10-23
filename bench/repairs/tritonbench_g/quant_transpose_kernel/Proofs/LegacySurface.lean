import bench.repairs.tritonbench_g.quant_transpose_kernel.Kernels

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource.LegacySurface
open Lean VeriTile.Triton
syntax "quantTranspose%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(quantTranspose%[$version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let pointer : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| A = A + (rm[:, None] * $((p.sam : Nat)) + rn[None, :] * $((p.san : Nat))))
      else
      `(tritonStmt| A = A + ($((p.sam : Nat)) * rm[:, None] + rn[None, :] * $((p.san : Nat))))
    `(fun (p : Parameters) =>
      let A : RegionName := ⟨p.x⟩
      let AbsmaxInv : RegionName := ⟨p.scale⟩
      let B : RegionName := ⟨p.output⟩
      ComputeKernel.mk [A, AbsmaxInv] [B] (triton {
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
        $pointer
        mask = (rm < $((p.M : Nat)))[:, None] & (rn < $((p.N : Nat)))[None, :]
        a = tl.load(A, mask=mask)
        absmax_inv = tl.load(AbsmaxInv)
        rm = pid_m * $(128) + tl.arange(0, $(128))
        rn = pid_n * $(128) + tl.arange(0, $(128))
        B = B + (rm[:, None] * $(1) + rn[None, :] * $((p.M : Nat)))
        mask = (rm < $((p.M : Nat)))[:, None] & (rn < $((p.N : Nat)))[None, :]
        output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
        tl.store(B, output, mask=mask)
      }).surfaceBody)

def kernel (version : Bool) : Parameters → ComputeKernel :=
  match version with
  | Bool.false => quantTranspose%[original]
  | Bool.true => quantTranspose%[commuted]

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource.LegacySurface
