import bench.tritonbench_g.seeded_dropout.Sources.Parameters
/-! Shared repaired seeded dropout and its integer-index candidate.
Dtype and seed-ABI cases specialize one source algorithm. The checked arange
adapter records source representability without changing the block size. -/
namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Triton

inductive SeedKind where
  | i32 | i64 | u64
  deriving DecidableEq

def SeedCarrier : SeedKind → Type
  | .i32 | .i64 => Int
  | .u64 => Nat

def seedValue : (kind : SeedKind) → SeedCarrier kind → Int
  | .i32,s | .i64,s => s
  | .u64,s => Int.ofNat s

open VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource (Storage)

def sourceFamily (storage : Storage) (kind : Random.IntegerType) (swapped : Bool) :
    RegionName → RegionName → Nat → ℝ →
      VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource.SeedCarrier kind →
      (B : Nat) → Random.ArangeSizeDomain B → ComputeKernel :=
  match storage,kind,swapped with
  | .fp32, .i32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .i32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .i64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .i64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .u32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .u32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .u64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp32, .u64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
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
  | .fp16, .i32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .i32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .i64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .i64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .u32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .u32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .u64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .u64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, output_ptr = tl.float16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .bf16, .i32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .i32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .i64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .i64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .u32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .u32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .u64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .u64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, output_ptr = tl.bfloat16
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .fp64, .i32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .i32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .i64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .i64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .u32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .u32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .u64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .fp64, .u64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float64, output_ptr = tl.float64
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float64), mask=mask)
  }
  | .f8e4, .i32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .i32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .i64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .i64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .u32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .u32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .u64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e4, .u64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e4nv, output_ptr = tl.float8e4nv
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e4nv), mask=mask)
  }
  | .f8e5, .i32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .i32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .i64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .i64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Int) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Int))
    tl.source_integer seed_arg = tl.int64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .u32, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .u32, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint32
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .u64, Bool.false =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_arg
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }
  | .f8e5, .u64, Bool.true =>
    fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : Nat) (BLOCK_SIZE : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) => triton {
    tl.ports inputs=[x_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float8e5, output_ptr = tl.float8e5
    seed_arg = $((seed : Nat))
    tl.source_integer seed_arg = tl.uint64
    BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE hBlock)
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE_arg * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed_arg, offsets)
    x_keep = random > $(p)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float8e5), mask=mask)
  }

private theorem fixed_block_domain : Random.ArangeSizeDomain 1024 :=
  ⟨by decide,by decide,10,by decide⟩

def selected (kind : SeedKind) (swapped : Bool) :
    RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => fun x y n p seed => sourceFamily .fp32 .i32 swapped x y n p seed 1024 fixed_block_domain
  | .i64 => fun x y n p seed => sourceFamily .fp32 .i64 swapped x y n p seed 1024 fixed_block_domain
  | .u64 => fun x y n p seed => sourceFamily .fp32 .u64 swapped x y n p seed 1024 fixed_block_domain

def kernel (kind : SeedKind) := selected kind Bool.false
def indexCommutedKernel (kind : SeedKind) := selected kind Bool.true
def gridCount (n : Nat) : Nat := (n+1023)/1024
end VeriTile.Bench.Repairs.SeededDropout
