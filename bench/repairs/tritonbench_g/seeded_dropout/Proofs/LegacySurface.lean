import bench.repairs.tritonbench_g.seeded_dropout.Kernels

namespace VeriTile.Bench.Repairs.SeededDropout.LegacySurface
open Lean VeriTile.Triton
syntax "checkedDropoutSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(checkedDropoutSource%[$seedDType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = BLOCK_SIZE_arg * pid)
      else `(tritonStmt| block_start = pid * BLOCK_SIZE_arg)
    let seedTy : TSyntax `term ← match seedDType with
      | `(tritonDType| tl.uint64) => `(Nat)
      | _ => `(Int)
    `(fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : $seedTy) =>
      ComputeKernel.mk [x_ptr] [output_ptr] (triton {
  tl.region x_ptr = tl.float32, output_ptr = tl.float32
  seed_arg = $((seed : $seedTy))
  tl.source_integer seed_arg = $seedDType
  BLOCK_SIZE_arg = $((1024 : Nat))
  pid = tl.program_id(axis=0)
  $blockStart
  offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  random = tl.rand(seed_arg, offsets)
  x_keep = random > $(p)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
}).surfaceBody)

def kernel (kind : SeedKind) : RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => checkedDropoutSource%[tl.int32, original]
  | .i64 => checkedDropoutSource%[tl.int64, original]
  | .u64 => checkedDropoutSource%[tl.uint64, original]

def indexCommutedKernel (kind : SeedKind) : RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => checkedDropoutSource%[tl.int32, indexCommuted]
  | .i64 => checkedDropoutSource%[tl.int64, indexCommuted]
  | .u64 => checkedDropoutSource%[tl.uint64, indexCommuted]

def originalKernel (kind : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier kind) : ComputeKernel :=
  kernel kind "x_ptr" "output_ptr" n p seed

def candidateKernel (kind : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier kind) : ComputeKernel :=
  indexCommutedKernel kind "x_ptr" "output_ptr" n p seed

def gridCount (n : Nat) : Nat := (n+1023)/1024
end VeriTile.Bench.Repairs.SeededDropout.LegacySurface
