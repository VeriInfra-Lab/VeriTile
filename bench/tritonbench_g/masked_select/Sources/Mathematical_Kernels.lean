import bench.tritonbench_g.masked_select.Sources.Selected
import VeriTile.Triton

/-! Source definitions only: the original seven statements and the retained typed
source materializations. The current whole-task public profile selects FP32;
legacy dtype definitions remain available without adding new verification work.
Host interpretation and mathematical targets live in Proofs/HostSource.lean. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect
open Lean VeriTile.Triton

namespace TypedCopy
syntax "maskedSelectTypedSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(maskedSelectTypedSource%[$storage:tritonDType]) =>
    `(fun (p : Parameters) =>
      let inp_ptr := p.input
      let out_ptr := p.output
      let select_mask_ptr := p.selection
      let prefix_sum_ptr := p.prefixSum
      ComputeKernel.mk [inp_ptr, select_mask_ptr.cast, prefix_sum_ptr.cast] [out_ptr]
      (triton {
        tl.region inp_ptr = $storage, out_ptr = $storage
        pid = tl.program_id(axis=0)
        offsets = pid * $(p.width) + tl.arange(0, $(p.width))
        mask = offsets < $(p.size)
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        select_mask = tl.load(select_mask_ptr + offsets,
          mask=mask, other=0.0).to(tl.int1)
        out_offset = tl.load(prefix_sum_ptr + offsets,
          mask=mask, other=0.0) - $(1)
        tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
      }).surfaceBody)

def source : Storage → Parameters → ComputeKernel
  | .fp16 => maskedSelectTypedSource%[tl.float16]
  | .bf16 => maskedSelectTypedSource%[tl.bfloat16]
  | .fp32 => VeriTile.Bench.Repairs.MaskedSelect.kernel
  | .fp64 => maskedSelectTypedSource%[tl.float64]
  | .f8e4 => maskedSelectTypedSource%[tl.float8e4nv]
  | .f8e5 => maskedSelectTypedSource%[tl.float8e5]
  | .int64 => maskedSelectTypedSource%[tl.int64]

end TypedCopy
namespace Whole
open TypedCopy
abbrev kernel := TypedCopy.source .fp32

structure Config where
  width : Nat
  warps : Nat
  deriving DecidableEq, Repr

def configs : List Config :=
  [4, 8, 16, 32].flatMap (fun w => [256, 512, 1024, 2048, 4096].map (fun b => ⟨b, w⟩))

def grid (n width : Nat) : Nat := (n + width - 1) / width

end Whole
end VeriTile.Bench.TritonBenchG.MaskedSelect
