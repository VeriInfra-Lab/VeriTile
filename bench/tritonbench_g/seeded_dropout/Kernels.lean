import bench.tritonbench_g.seeded_dropout.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.IntegerCallArguments
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

def packedSeed : (kind : Random.IntegerType) →
    VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource.SeedCarrier kind → IntegerArgument
  | .i32, s => .i32 s
  | .i64, s => .i64 s
  | .u32, s => .u32 s
  | .u64, s => .u64 s

def callSignature (storage : Storage) : CallSignature where
  pointers := [("x_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("output_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  scalars := [("n_elements",.integer),("p",.fp32),("seed",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr"]
  writePorts := ["output_ptr"]

def runtimeKernel (storage : Storage) (x_ptr output_ptr : RegionName)
    (n_elements : Nat) (seed : IntegerArgument) (BLOCK_SIZE : Nat)
    (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) : ComputeKernel :=
  with_triton_call (callSignature storage) call_args [n_elements := n_elements]
    integer_args [seed := seed] arange_args [BLOCK_SIZE := hBlock] in triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def runtimeCandidateKernel (storage : Storage) (x_ptr output_ptr : RegionName)
    (n_elements : Nat) (seed : IntegerArgument) (BLOCK_SIZE : Nat)
    (hBlock : Random.ArangeSizeDomain BLOCK_SIZE) : ComputeKernel :=
  with_triton_call (callSignature storage) call_args [n_elements := n_elements]
    integer_args [seed := seed] arange_args [BLOCK_SIZE := hBlock] in triton {
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- Bind the host probability once in the FP32 argument channel. It remains
an ordinary runtime register inside the copied body, including in FP64 payload
profiles; no surrounding operation may reinterpret it as a weak literal. -/
noncomputable def probabilityArgument (p : ℝ) : ScalarArgument := ScalarArgument.fp32 "p" p

noncomputable def rawSource (storage : Storage) (X Y : RegionName) (n : Nat)
    (p : ℝ) (seed : IntegerArgument) (B : Nat) (legal : Random.ArangeSizeDomain B) : ComputeKernel :=
  (runtimeKernel storage X Y n seed B legal).withScalarArguments [probabilityArgument p]

noncomputable def rawCandidate (storage : Storage) (X Y : RegionName) (n : Nat)
    (p : ℝ) (seed : IntegerArgument) (B : Nat) (legal : Random.ArangeSizeDomain B) : ComputeKernel :=
  (runtimeCandidateKernel storage X Y n seed B legal).withScalarArguments [probabilityArgument p]

/-- All retained storage/seed ABI profiles specialize the same copied body.
The wrapper's fixed block domain and signed/unsigned seed range stay separate. -/
noncomputable def sourceFamily (storage : Storage) (kind : Random.IntegerType) (swapped : Bool) :
    RegionName → RegionName → Nat → ℝ →
      VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource.SeedCarrier kind →
      (B : Nat) → Random.ArangeSizeDomain B → ComputeKernel :=
  fun x y n p seed B hB =>
    match swapped with
    | Bool.true => rawCandidate storage x y n p (packedSeed kind seed) B hB
    | Bool.false => rawSource storage x y n p (packedSeed kind seed) B hB

private theorem fixed_block_domain : Random.ArangeSizeDomain 1024 :=
  ⟨by decide,by decide,10,by decide⟩

noncomputable def selected (kind : SeedKind) (swapped : Bool) :
    RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => fun x y n p seed => sourceFamily .fp32 .i32 swapped x y n p seed 1024 fixed_block_domain
  | .i64 => fun x y n p seed => sourceFamily .fp32 .i64 swapped x y n p seed 1024 fixed_block_domain
  | .u64 => fun x y n p seed => sourceFamily .fp32 .u64 swapped x y n p seed 1024 fixed_block_domain

noncomputable def kernel (kind : SeedKind) := selected kind Bool.false
noncomputable def indexCommutedKernel (kind : SeedKind) := selected kind Bool.true
def gridCount (n : Nat) : Nat := (n+1023)/1024
end VeriTile.Bench.Repairs.SeededDropout
