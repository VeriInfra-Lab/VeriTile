import VeriTile.Triton.DSL

/-! Checked FP32 source and its existing index variant, fixed BLOCK_SIZE=1024. Runtime p,
canonical Philox-10, omitted load padding, eager division and masked store
are retained. i32/i64/u64 are the three normal Python-int JIT branches. -/
namespace VeriTile.Bench.Repairs.SeededDropout
open Lean VeriTile.Triton

inductive SeedKind where
  | i32 | i64 | u64
  deriving DecidableEq

def SeedCarrier : SeedKind → Type
  | .i32 | .i64 => Int
  | .u64 => Nat

def seedValue : (kind : SeedKind) → SeedCarrier kind → Int
  | .i32, s | .i64, s => s
  | .u64, s => Int.ofNat s

def kernel (kind : SeedKind) : RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : Int) =>
    triton {
      tl.region x_ptr = tl.float32, output_ptr = tl.float32
      seed_arg = $((seed : Int))
      tl.source_integer seed_arg = tl.int32
      BLOCK_SIZE_arg = $((1024 : Nat))
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE_arg
      offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
      mask = offsets < $(n_elements)
      x = tl.load(x_ptr + offsets, mask=mask)
      random = tl.rand(seed_arg, offsets)
      x_keep = random > $(p)
      output = tl.where(x_keep, x / (1 - $(p)), 0.0)
      tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
    }
  | .i64 => fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : Int) =>
    triton {
      tl.region x_ptr = tl.float32, output_ptr = tl.float32
      seed_arg = $((seed : Int))
      tl.source_integer seed_arg = tl.int64
      BLOCK_SIZE_arg = $((1024 : Nat))
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE_arg
      offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
      mask = offsets < $(n_elements)
      x = tl.load(x_ptr + offsets, mask=mask)
      random = tl.rand(seed_arg, offsets)
      x_keep = random > $(p)
      output = tl.where(x_keep, x / (1 - $(p)), 0.0)
      tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
    }
  | .u64 => fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : Nat) =>
    triton {
      tl.region x_ptr = tl.float32, output_ptr = tl.float32
      seed_arg = $((seed : Nat))
      tl.source_integer seed_arg = tl.uint64
      BLOCK_SIZE_arg = $((1024 : Nat))
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE_arg
      offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
      mask = offsets < $(n_elements)
      x = tl.load(x_ptr + offsets, mask=mask)
      random = tl.rand(seed_arg, offsets)
      x_keep = random > $(p)
      output = tl.where(x_keep, x / (1 - $(p)), 0.0)
      tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
    }

def indexCommutedKernel (kind : SeedKind) : RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : Int) =>
    triton {
      tl.region x_ptr = tl.float32, output_ptr = tl.float32
      seed_arg = $((seed : Int))
      tl.source_integer seed_arg = tl.int32
      BLOCK_SIZE_arg = $((1024 : Nat))
      pid = tl.program_id(axis=0)
      block_start = BLOCK_SIZE_arg * pid
      offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
      mask = offsets < $(n_elements)
      x = tl.load(x_ptr + offsets, mask=mask)
      random = tl.rand(seed_arg, offsets)
      x_keep = random > $(p)
      output = tl.where(x_keep, x / (1 - $(p)), 0.0)
      tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
    }
  | .i64 => fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : Int) =>
    triton {
      tl.region x_ptr = tl.float32, output_ptr = tl.float32
      seed_arg = $((seed : Int))
      tl.source_integer seed_arg = tl.int64
      BLOCK_SIZE_arg = $((1024 : Nat))
      pid = tl.program_id(axis=0)
      block_start = BLOCK_SIZE_arg * pid
      offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
      mask = offsets < $(n_elements)
      x = tl.load(x_ptr + offsets, mask=mask)
      random = tl.rand(seed_arg, offsets)
      x_keep = random > $(p)
      output = tl.where(x_keep, x / (1 - $(p)), 0.0)
      tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
    }
  | .u64 => fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ) (seed : Nat) =>
    triton {
      tl.region x_ptr = tl.float32, output_ptr = tl.float32
      seed_arg = $((seed : Nat))
      tl.source_integer seed_arg = tl.uint64
      BLOCK_SIZE_arg = $((1024 : Nat))
      pid = tl.program_id(axis=0)
      block_start = BLOCK_SIZE_arg * pid
      offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
      mask = offsets < $(n_elements)
      x = tl.load(x_ptr + offsets, mask=mask)
      random = tl.rand(seed_arg, offsets)
      x_keep = random > $(p)
      output = tl.where(x_keep, x / (1 - $(p)), 0.0)
      tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
    }

def originalKernel (kind : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier kind) : ComputeKernel :=
  kernel kind "x_ptr" "output_ptr" n p seed

def candidateKernel (kind : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier kind) : ComputeKernel :=
  indexCommutedKernel kind "x_ptr" "output_ptr" n p seed

def gridCount (n : Nat) : Nat := (n+1023)/1024
end VeriTile.Bench.Repairs.SeededDropout
