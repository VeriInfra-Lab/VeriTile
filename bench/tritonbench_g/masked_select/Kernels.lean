import VeriTile.Triton

/-! Source definitions only: the original seven statements and the retained typed
source materializations. The current whole-task public profile selects FP32;
legacy dtype definitions remain available without adding new verification work.
Host interpretation and mathematical targets live in Proofs/HostSource.lean. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect
open Lean VeriTile.Triton

/-- Faithful transcription of `masked_select.py`'s `masked_select_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- `select_mask_ptr` and `prefix_sum_ptr` are typed Lean regions so their
  `tl.load` calls do not need extra `dtype=` kwargs. -/
def masked_select_kernel
    (inp_ptr : RegionName) (select_mask_ptr : Region .bool)
    (prefix_sum_ptr : Region .nat) (out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  offsets = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
  select_mask = tl.load(select_mask_ptr + offsets,
    mask=mask, other=0.0).to(tl.int1)
  out_offset = tl.load(prefix_sum_ptr + offsets,
    mask=mask, other=0.0) - $(1)
  tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
}

namespace TypedCopy
inductive Storage where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5 | int64
  deriving DecidableEq, Repr

def Storage.tag : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 | .fp64 => .real
  | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .int64 => .int

structure Parameters where
  (input output : RegionName)
  selection : Region .bool
  prefixSum : Region .nat
  (size width : Nat)

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
  | .fp32 => maskedSelectTypedSource%[tl.float32]
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
