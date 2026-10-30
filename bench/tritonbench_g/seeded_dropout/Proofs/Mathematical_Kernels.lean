/-
Source-audit target: _seeded_dropout
Source-audit Lean: seededDropoutSource%
Source-audit args: storage=tl.float32, seedDType=tl.int32, version=original
Source-audit materializations: {"source_sha256":"2fd4f4e6aff35d5720156b9cf046f2d112173a431d886106a6441382f63ab570","lean":{"source":"seededDropoutSource%","args":{"storage":"tl.float32","seedDType":"tl.int32","version":"original"}},"regions":{"x_ptr":"tl.float32","output_ptr":"tl.float32"},"bindings":[{"python":"seed","local":"seed_arg","kind":"runtime_integer","dtype":"tl.int32"},{"python":"BLOCK_SIZE","local":"BLOCK_SIZE_arg","kind":"constexpr_identity","adapter":"_root_.VeriTile.Triton.Random.i32ArangeSize","adapter_sha256":"61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04","proof":{"function":"RandomDomain.arange_domain","binder":"hBlock"}}]}
-/
import VeriTile.Triton.DSL
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_RandomDomain

/-! Source candidate for pinned TritonBench-G `_seeded_dropout`.
The runtime seed and p, omitted load padding, fixed canonical rand, eager
division and final masked store remain. The source constexpr adapter records
the already proved arange domain without changing B. The explicit destination
cast represents native store conversion. The candidate commutes only the
integer block-start multiplication. These formal storage profiles do not by
themselves assert native admission or IEEE correctness. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource
open Lean VeriTile.Triton

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq

def SeedCarrier : Random.IntegerType → Type
  | .i32 | .i64 => Int
  | .u32 | .u64 => Nat

def seedInt (kind : Random.IntegerType) (seed : SeedCarrier kind) : Int :=
  match kind, seed with
  | .i32, s | .i64, s => s
  | .u32, s | .u64, s => Int.ofNat s

syntax "seededDropoutSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(seededDropoutSource%[$storage:tritonDType, $seedDType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let seedTy : TSyntax `term ←
      match seedDType with
      | `(tritonDType| tl.int32) | `(tritonDType| tl.int64) => `(Int)
      | `(tritonDType| tl.uint32) | `(tritonDType| tl.uint64) => `(Nat)
      | _ => Macro.throwError "expected a supported source integer seed type"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = BLOCK_SIZE_arg * pid)
      else `(tritonStmt| block_start = pid * BLOCK_SIZE_arg)
    `(fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : $seedTy) (BLOCK_SIZE : Nat) (hBlock : SourceDomain.LegalBlock BLOCK_SIZE) =>
      ComputeKernel.mk [x_ptr] [output_ptr] (triton {
  tl.region x_ptr = $storage, output_ptr = $storage
  seed_arg = $((seed : $seedTy))
  tl.source_integer seed_arg = $seedDType
  BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE (RandomDomain.arange_domain hBlock))
  pid = tl.program_id(axis=0)
  $blockStart
  offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  random = tl.rand(seed_arg, offsets)
  x_keep = random > $(p)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, (output).to($storage:tritonDType), mask=mask)
}).surfaceBody)

abbrev sourceType (seedKind : Random.IntegerType) :=
  RegionName → RegionName → Nat → ℝ → SeedCarrier seedKind →
    (B : Nat) → SourceDomain.LegalBlock B → ComputeKernel

def source (storage : Storage) (seedKind : Random.IntegerType) (swapped : Bool) :
    sourceType seedKind :=
  match storage, seedKind, swapped with
  | .fp32, .i32, Bool.false => seededDropoutSource%[tl.float32, tl.int32, original]
  | .fp32, .i32, Bool.true => seededDropoutSource%[tl.float32, tl.int32, indexCommuted]
  | .fp32, .i64, Bool.false => seededDropoutSource%[tl.float32, tl.int64, original]
  | .fp32, .i64, Bool.true => seededDropoutSource%[tl.float32, tl.int64, indexCommuted]
  | .fp32, .u32, Bool.false => seededDropoutSource%[tl.float32, tl.uint32, original]
  | .fp32, .u32, Bool.true => seededDropoutSource%[tl.float32, tl.uint32, indexCommuted]
  | .fp32, .u64, Bool.false => seededDropoutSource%[tl.float32, tl.uint64, original]
  | .fp32, .u64, Bool.true => seededDropoutSource%[tl.float32, tl.uint64, indexCommuted]
  | .fp16, .i32, Bool.false => seededDropoutSource%[tl.float16, tl.int32, original]
  | .fp16, .i32, Bool.true => seededDropoutSource%[tl.float16, tl.int32, indexCommuted]
  | .fp16, .i64, Bool.false => seededDropoutSource%[tl.float16, tl.int64, original]
  | .fp16, .i64, Bool.true => seededDropoutSource%[tl.float16, tl.int64, indexCommuted]
  | .fp16, .u32, Bool.false => seededDropoutSource%[tl.float16, tl.uint32, original]
  | .fp16, .u32, Bool.true => seededDropoutSource%[tl.float16, tl.uint32, indexCommuted]
  | .fp16, .u64, Bool.false => seededDropoutSource%[tl.float16, tl.uint64, original]
  | .fp16, .u64, Bool.true => seededDropoutSource%[tl.float16, tl.uint64, indexCommuted]
  | .bf16, .i32, Bool.false => seededDropoutSource%[tl.bfloat16, tl.int32, original]
  | .bf16, .i32, Bool.true => seededDropoutSource%[tl.bfloat16, tl.int32, indexCommuted]
  | .bf16, .i64, Bool.false => seededDropoutSource%[tl.bfloat16, tl.int64, original]
  | .bf16, .i64, Bool.true => seededDropoutSource%[tl.bfloat16, tl.int64, indexCommuted]
  | .bf16, .u32, Bool.false => seededDropoutSource%[tl.bfloat16, tl.uint32, original]
  | .bf16, .u32, Bool.true => seededDropoutSource%[tl.bfloat16, tl.uint32, indexCommuted]
  | .bf16, .u64, Bool.false => seededDropoutSource%[tl.bfloat16, tl.uint64, original]
  | .bf16, .u64, Bool.true => seededDropoutSource%[tl.bfloat16, tl.uint64, indexCommuted]
  | .fp64, .i32, Bool.false => seededDropoutSource%[tl.float64, tl.int32, original]
  | .fp64, .i32, Bool.true => seededDropoutSource%[tl.float64, tl.int32, indexCommuted]
  | .fp64, .i64, Bool.false => seededDropoutSource%[tl.float64, tl.int64, original]
  | .fp64, .i64, Bool.true => seededDropoutSource%[tl.float64, tl.int64, indexCommuted]
  | .fp64, .u32, Bool.false => seededDropoutSource%[tl.float64, tl.uint32, original]
  | .fp64, .u32, Bool.true => seededDropoutSource%[tl.float64, tl.uint32, indexCommuted]
  | .fp64, .u64, Bool.false => seededDropoutSource%[tl.float64, tl.uint64, original]
  | .fp64, .u64, Bool.true => seededDropoutSource%[tl.float64, tl.uint64, indexCommuted]
  | .f8e4, .i32, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.int32, original]
  | .f8e4, .i32, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e4, .i64, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.int64, original]
  | .f8e4, .i64, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e4, .u32, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.uint32, original]
  | .f8e4, .u32, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.uint32, indexCommuted]
  | .f8e4, .u64, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.uint64, original]
  | .f8e4, .u64, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.uint64, indexCommuted]
  | .f8e5, .i32, Bool.false => seededDropoutSource%[tl.float8e5, tl.int32, original]
  | .f8e5, .i32, Bool.true => seededDropoutSource%[tl.float8e5, tl.int32, indexCommuted]
  | .f8e5, .i64, Bool.false => seededDropoutSource%[tl.float8e5, tl.int64, original]
  | .f8e5, .i64, Bool.true => seededDropoutSource%[tl.float8e5, tl.int64, indexCommuted]
  | .f8e5, .u32, Bool.false => seededDropoutSource%[tl.float8e5, tl.uint32, original]
  | .f8e5, .u32, Bool.true => seededDropoutSource%[tl.float8e5, tl.uint32, indexCommuted]
  | .f8e5, .u64, Bool.false => seededDropoutSource%[tl.float8e5, tl.uint64, original]
  | .f8e5, .u64, Bool.true => seededDropoutSource%[tl.float8e5, tl.uint64, indexCommuted]

end VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource
