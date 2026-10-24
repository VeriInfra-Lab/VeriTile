/-
Source-audit target: softmax_kernel
Source-audit Lean: softmaxSource%
Source-audit args: inputType=tl.float32, outputType=tl.float32, version=original
This selected source view is checked separately from the complete dtype family.
-/

import VeriTile.Triton.DSL

/-!
# `triton_softmax` — strict per-kernel correctness

`softmax_kernel` is a row-wise numerically-stable softmax: program `row_idx`
loads one row of width `BLOCK_SIZE` (masked by `arange < n_cols`, with
`other=-inf`), subtracts the row max, exponentiates, normalizes by the row sum,
and stores the row back.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`softmax_kernel[(n_rows,)](...)`, the grid size, the
host-side `BLOCK_SIZE = min(next_power_of_2(n_cols), 1024)` choice, the row
strides, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because `row_idx` (= the
program id) is universally quantified, the per-program statement covers every
row of the grid.

## Proof architecture

```
softmax_typed_real_correct                    ← PUBLIC typedIO ⊨ softmaxSpec
  ├─ typed_flat_exec                         complete typed-source projection
  └─ softmax_kernel_real_io                  internal mathematical IO bridge
       ├─ softmax_kernel_flattenOk           bridge fragment membership
       ├─ softmax_kernel_traceSafe           per-execution lane-wise safety walk
       └─ softmax_kernel_region_run          termination, active values and frame
```

The public contract is the masked Hoare-triple combinator `typedIO … ⊨ softmaxSpec`
(`MaskedKernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program id all of whose *active* lanes (`j < n_cols`) are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane holds
`softmaxSpec n_cols BLOCK_SIZE xs j`, and every other memory cell is unchanged.

The spec `softmaxSpec` is the exact stable softmax over the active prefix:
`reduceMax` over the masked row, lane-wise `exp(row - rowMax)`, `reduceSum` for
the denominator, then the normalizing division.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the approximate `tl.exp`
is modeled as exact `WithBot.realExp`. The row reduction (`tl.max` / `tl.sum`)
runs over the full padded `BLOCK_SIZE` tile, with masked lanes set to `⊥`
matching `other=-float("inf")`, so padding lanes do not contribute to the max
or sum. The spec references `Tile.reduceMax` / `Tile.reduceSum` /
`WithBot.realExp` directly, not `VeriTile.Triton.Math.*`. No output/input
disjointness is assumed in the region model: the row is read into registers
before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax

open Lean VeriTile.Triton

/-- Faithful 1:1 transcription of `triton_softmax.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
-/
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(axis=0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  out_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  row = tl.load(row_start_ptr + tl.arange(0, $(BLOCK_SIZE)),
    mask=tl.arange(0, $(BLOCK_SIZE)) < $(n_cols), other=-float("inf"))
  row_max = tl.max(row, axis=0)
  numerator = tl.exp(row - row_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(out_row_start_ptr + tl.arange(0, $(BLOCK_SIZE)),
    softmax_output, mask=tl.arange(0, $(BLOCK_SIZE)) < $(n_cols))
}


/-- Floating input/output storage is independent at the JIT boundary. The
wrapper correlates them through empty_like. Reductions promote fp16/bf16/fp8
to fp32; fp64 remains fp64. Explicit destination casts expose implicit stores. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

syntax "softmaxSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(softmaxSource%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowStart : Lean.TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| row_start_ptr = input_ptr + row_idx * $(input_row_stride))
      else `(tritonStmt| row_start_ptr = input_ptr + $(input_row_stride) * row_idx)
    let sameCompute : Bool := match inputType, outputType with
      | `(tritonDType| tl.float64), `(tritonDType| tl.float64) => Bool.true
      | `(tritonDType| tl.float64), _ => Bool.false
      | _, `(tritonDType| tl.float32) => Bool.true
      | _, _ => Bool.false
    let outputValue : Lean.TSyntax `tritonExpr ←
      if sameCompute then `(tritonExpr| softmax_output)
      else `(tritonExpr| (softmax_output).to($outputType:tritonDType))
    `(fun (output_ptr input_ptr : RegionName)
        (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [input_ptr] [output_ptr] (triton {
  tl.region input_ptr = $inputType, output_ptr = $outputType
  row_idx = tl.program_id(axis=0)
  $rowStart
  out_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  row = tl.load(row_start_ptr + tl.arange(0, $(BLOCK_SIZE)),
    mask=tl.arange(0, $(BLOCK_SIZE)) < $(n_cols), other=-float("inf"))
  row_max = tl.max(row, axis=0)
  numerator = tl.exp(row - row_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(out_row_start_ptr + tl.arange(0, $(BLOCK_SIZE)),
    $outputValue, mask=tl.arange(0, $(BLOCK_SIZE)) < $(n_cols))
}).surfaceBody)

def typedKernel (input output : Storage) (commuted : Bool) : Source :=
  match input, output, commuted with
  | .fp32, .fp32, Bool.false => softmaxSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmaxSource%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp64, Bool.false => softmaxSource%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => softmaxSource%[tl.float32, tl.float64, indexCommuted]
  | .fp32, .fp16, Bool.false => softmaxSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => softmaxSource%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => softmaxSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => softmaxSource%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp32, .f8e4, Bool.false => softmaxSource%[tl.float32, tl.float8e4nv, original]
  | .fp32, .f8e4, Bool.true => softmaxSource%[tl.float32, tl.float8e4nv, indexCommuted]
  | .fp32, .f8e5, Bool.false => softmaxSource%[tl.float32, tl.float8e5, original]
  | .fp32, .f8e5, Bool.true => softmaxSource%[tl.float32, tl.float8e5, indexCommuted]
  | .fp64, .fp32, Bool.false => softmaxSource%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => softmaxSource%[tl.float64, tl.float32, indexCommuted]
  | .fp64, .fp64, Bool.false => softmaxSource%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => softmaxSource%[tl.float64, tl.float64, indexCommuted]
  | .fp64, .fp16, Bool.false => softmaxSource%[tl.float64, tl.float16, original]
  | .fp64, .fp16, Bool.true => softmaxSource%[tl.float64, tl.float16, indexCommuted]
  | .fp64, .bf16, Bool.false => softmaxSource%[tl.float64, tl.bfloat16, original]
  | .fp64, .bf16, Bool.true => softmaxSource%[tl.float64, tl.bfloat16, indexCommuted]
  | .fp64, .f8e4, Bool.false => softmaxSource%[tl.float64, tl.float8e4nv, original]
  | .fp64, .f8e4, Bool.true => softmaxSource%[tl.float64, tl.float8e4nv, indexCommuted]
  | .fp64, .f8e5, Bool.false => softmaxSource%[tl.float64, tl.float8e5, original]
  | .fp64, .f8e5, Bool.true => softmaxSource%[tl.float64, tl.float8e5, indexCommuted]
  | .fp16, .fp32, Bool.false => softmaxSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmaxSource%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp64, Bool.false => softmaxSource%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => softmaxSource%[tl.float16, tl.float64, indexCommuted]
  | .fp16, .fp16, Bool.false => softmaxSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => softmaxSource%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => softmaxSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => softmaxSource%[tl.float16, tl.bfloat16, indexCommuted]
  | .fp16, .f8e4, Bool.false => softmaxSource%[tl.float16, tl.float8e4nv, original]
  | .fp16, .f8e4, Bool.true => softmaxSource%[tl.float16, tl.float8e4nv, indexCommuted]
  | .fp16, .f8e5, Bool.false => softmaxSource%[tl.float16, tl.float8e5, original]
  | .fp16, .f8e5, Bool.true => softmaxSource%[tl.float16, tl.float8e5, indexCommuted]
  | .bf16, .fp32, Bool.false => softmaxSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmaxSource%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp64, Bool.false => softmaxSource%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => softmaxSource%[tl.bfloat16, tl.float64, indexCommuted]
  | .bf16, .fp16, Bool.false => softmaxSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => softmaxSource%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => softmaxSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => softmaxSource%[tl.bfloat16, tl.bfloat16, indexCommuted]
  | .bf16, .f8e4, Bool.false => softmaxSource%[tl.bfloat16, tl.float8e4nv, original]
  | .bf16, .f8e4, Bool.true => softmaxSource%[tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .bf16, .f8e5, Bool.false => softmaxSource%[tl.bfloat16, tl.float8e5, original]
  | .bf16, .f8e5, Bool.true => softmaxSource%[tl.bfloat16, tl.float8e5, indexCommuted]
  | .f8e4, .fp32, Bool.false => softmaxSource%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => softmaxSource%[tl.float8e4nv, tl.float32, indexCommuted]
  | .f8e4, .fp64, Bool.false => softmaxSource%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => softmaxSource%[tl.float8e4nv, tl.float64, indexCommuted]
  | .f8e4, .fp16, Bool.false => softmaxSource%[tl.float8e4nv, tl.float16, original]
  | .f8e4, .fp16, Bool.true => softmaxSource%[tl.float8e4nv, tl.float16, indexCommuted]
  | .f8e4, .bf16, Bool.false => softmaxSource%[tl.float8e4nv, tl.bfloat16, original]
  | .f8e4, .bf16, Bool.true => softmaxSource%[tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, .f8e4, Bool.false => softmaxSource%[tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, .f8e4, Bool.true => softmaxSource%[tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e4, .f8e5, Bool.false => softmaxSource%[tl.float8e4nv, tl.float8e5, original]
  | .f8e4, .f8e5, Bool.true => softmaxSource%[tl.float8e4nv, tl.float8e5, indexCommuted]
  | .f8e5, .fp32, Bool.false => softmaxSource%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => softmaxSource%[tl.float8e5, tl.float32, indexCommuted]
  | .f8e5, .fp64, Bool.false => softmaxSource%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => softmaxSource%[tl.float8e5, tl.float64, indexCommuted]
  | .f8e5, .fp16, Bool.false => softmaxSource%[tl.float8e5, tl.float16, original]
  | .f8e5, .fp16, Bool.true => softmaxSource%[tl.float8e5, tl.float16, indexCommuted]
  | .f8e5, .bf16, Bool.false => softmaxSource%[tl.float8e5, tl.bfloat16, original]
  | .f8e5, .bf16, Bool.true => softmaxSource%[tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e5, .f8e4, Bool.false => softmaxSource%[tl.float8e5, tl.float8e4nv, original]
  | .f8e5, .f8e4, Bool.true => softmaxSource%[tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, .f8e5, Bool.false => softmaxSource%[tl.float8e5, tl.float8e5, original]
  | .f8e5, .f8e5, Bool.true => softmaxSource%[tl.float8e5, tl.float8e5, indexCommuted]

end VeriTile.Bench.TritonBenchG.TritonSoftmax
