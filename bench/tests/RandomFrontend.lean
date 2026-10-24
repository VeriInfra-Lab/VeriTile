import VeriTile.Triton.DSL

/-! Actual DSL regressions for the closed random primitive. These check the
source-profile and precision boundary; core numerical vectors live separately.
No native/GPU correctness claim is made by these elaboration checks. -/
namespace VeriTile.Bench.Tests.RandomFrontend
open VeriTile.Triton

private def computeBody : ComputeKernel → List ComputeStmt
  | .mk _ _ body => body

private def finalRandProfile? (k : ComputeKernel) : Option Random.SourceProfile :=
  match (computeBody k).reverse.head? with
  | some (.assign _ _ _ (.compute (.alg .fp32 (.rand cfg _ _)))) => some cfg
  | _ => none

private def finalShape? (k : ComputeKernel) : Option TileShape :=
  match (computeBody k).reverse.head? with
  | some (.assign _ shape _ _) => some shape
  | _ => none

def seedI32 (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

def seedI64 (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int64
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

def seedU32 (seed : Nat) : ComputeKernel := triton {
  seed_arg = $((seed : Nat))
  tl.source_integer seed_arg = tl.uint32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

def seedU64 (seed : Nat) : ComputeKernel := triton {
  seed_arg = $((seed : Nat))
  tl.source_integer seed_arg = tl.uint64
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

theorem seedI32_profile (seed : Int) : finalRandProfile? (seedI32 seed) = some ⟨.i32, .i32, 10⟩ := rfl
example (seed : Int) : finalRandProfile? (seedI64 seed) = some ⟨.i64, .i32, 10⟩ := rfl
example (seed : Nat) : finalRandProfile? (seedU32 seed) = some ⟨.u32, .i32, 10⟩ := rfl
theorem seedU64_profile (seed : Nat) : finalRandProfile? (seedU64 seed) = some ⟨.u64, .i32, 10⟩ := rfl

def zeroPositional (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets, 0)
}

def zeroKeyword (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets, n_rounds=0)
}

theorem zeroRounds_profile (seed : Int) : finalRandProfile? (zeroPositional seed) = some ⟨.i32, .i32, 0⟩ := rfl
example (seed : Int) : zeroPositional seed = zeroKeyword seed := rfl

def symbolicRounds (seed : Int) (rounds : Nat) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets,
    n_rounds=$(_root_.VeriTile.Triton.Random.roundsConstexpr rounds))
}

example (seed : Int) (rounds : Nat) :
    finalRandProfile? (symbolicRounds seed rounds) = some ⟨.i32, .i32, rounds⟩ := rfl

def offsetsI64 (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int64
  offsets = tl.arange(0, 32).to(tl.int64)
  random = tl.rand(seed_arg, offsets)
}

example (seed : Int) : finalRandProfile? (offsetsI64 seed) = some ⟨.i64, .i64, 10⟩ := rfl

def symbolicBlock (seed : Int) (B : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  BLOCK_SIZE = $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB)
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  random = tl.rand(seed_arg, offsets)
}

example (seed : Int) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    finalRandProfile? (symbolicBlock seed B hB) = some ⟨.i32, .i32, 10⟩ := rfl

theorem symbolicBlock_shape (seed : Int) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    finalShape? (symbolicBlock seed B hB) = some [B] := rfl

-- The ordinary assignment and shape still carry the original symbolic B.
example (B : Nat) (hB : Random.ArangeSizeDomain B) :
    Random.i32ArangeSize B hB = B := rfl

def copyProfile (seed : Int) : ComputeKernel := triton {
  original = $((seed : Int))
  tl.source_integer original = tl.int64
  copied = original
  offsets = tl.arange(0, 32)
  random = tl.rand(copied, offsets)
}

example (seed : Int) : finalRandProfile? (copyProfile seed) = some ⟨.i64, .i32, 10⟩ := rfl

-- Control-flow joins retain only profiles established in both branches.
def branchSameProfile (left right : Int) (choose : Bool) : ComputeKernel := triton {
  left_seed = $((left : Int))
  tl.source_integer left_seed = tl.int64
  right_seed = $((right : Int))
  tl.source_integer right_seed = tl.int64
  if $((choose : Bool)) {
    seed_arg = left_seed
  } else {
    seed_arg = right_seed
  }
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

def tlBranchSameProfile (left right : Int) (choose : Bool) : ComputeKernel := triton {
  left_seed = $((left : Int))
  tl.source_integer left_seed = tl.int32
  right_seed = $((right : Int))
  tl.source_integer right_seed = tl.int32
  tl.if $((choose : Bool)) {
    seed_arg = left_seed
  } else {
    seed_arg = right_seed
  }
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

theorem branchSame_profile (left right : Int) (choose : Bool) :
    finalRandProfile? (branchSameProfile left right choose) = some ⟨.i64, .i32, 10⟩ := rfl
example (left right : Int) (choose : Bool) :
    finalRandProfile? (tlBranchSameProfile left right choose) = some ⟨.i32, .i32, 10⟩ := rfl

-- Both argument loads must remain visible to the kernel's region metadata.
def signedLoads (seed_mem offset_mem : Region .int) : ComputeKernel := triton {
  tl.region seed_mem = tl.int64, offset_mem = tl.int64
  random = tl.rand(tl.load(seed_mem + 0), tl.load(offset_mem + tl.arange(0, 32)))
}

def signed32Loads (seed_mem offset_mem : Region .int) : ComputeKernel := triton {
  tl.region seed_mem = tl.int32, offset_mem = tl.int32
  random = tl.rand(tl.load(seed_mem + 0), tl.load(offset_mem + tl.arange(0, 32)))
}

def unsignedLoads (seed_mem offset_mem : Region .nat) : ComputeKernel := triton {
  tl.region seed_mem = tl.uint32, offset_mem = tl.uint64
  random = tl.rand(tl.load(seed_mem + 0), tl.load(offset_mem + tl.arange(0, 32)))
}

example (a b : Region .int) : finalRandProfile? (signedLoads a b) = some ⟨.i64, .i64, 10⟩ := rfl
theorem signed32Loads_profile (a b : Region .int) :
    finalRandProfile? (signed32Loads a b) = some ⟨.i32, .i32, 10⟩ := rfl
example (a b : Region .nat) : finalRandProfile? (unsignedLoads a b) = some ⟨.u32, .u64, 10⟩ := rfl
example (a b : Region .int) : (signedLoads a b).inputs.length = 2 := rfl
theorem unsignedLoads_inputs (a b : Region .nat) : (unsignedLoads a b).inputs.length = 2 := rfl

def promoted (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets).to(tl.float64)
}

private def isRand {dtype shape} : Op dtype shape → Bool
  | .rand _ _ _ => Bool.true
  | _ => Bool.false

private def isScopedFP32Rand {dtype shape} : Op dtype shape → Bool
  | .withPrecision .real .fp32 child => isRand child
  | _ => Bool.false

private def isCastOfScopedFP32Rand {dtype shape} : Op dtype shape → Bool
  | .castFloat .real .real child => isScopedFP32Rand child
  | _ => Bool.false

private def hasFixedFP32InsideFP64 (k : ComputeKernel) : Bool :=
  match (computeBody k).reverse.head? with
  | some (.assign _ _ _ (.compute (.alg .fp64 op))) => isCastOfScopedFP32Rand op
  | _ => Bool.false

theorem promoted_retainsFP32 (seed : Int) :
    hasFixedFP32InsideFP64 (promoted seed) = Bool.true := rfl

/-- error: tl.rand: seed has no supported source integer profile; annotate a direct external scalar binding with tl.source_integer -/
#guard_msgs in
def missingSeedProfile (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.rand: offsets have no supported source integer profile; unknown symbolic sizes need a checked rooted i32ArangeSize adapter -/
#guard_msgs in
def unknownBlock (seed : Int) (B : Nat) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, $((B : Nat)))
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.rand: seed has no supported source integer profile; annotate a direct external scalar binding with tl.source_integer -/
#guard_msgs in
def reassignClears (seed next : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  seed_arg = $((next : Int))
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.source_integer: expected a preceding direct external Int/Nat scalar binding; computed values and aliases cannot acquire an ABI profile -/
#guard_msgs in
def computedAnnotation (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  other = seed_arg + 1
  tl.source_integer other = tl.int32
  offsets = tl.arange(0, 32)
  random = tl.rand(other, offsets)
}

/-- error: tl.source_integer: expected a preceding direct external Int/Nat scalar binding; computed values and aliases cannot acquire an ABI profile -/
#guard_msgs in
def aliasCannotManufactureProfile (seed : Int) : ComputeKernel := triton {
  original = $((seed : Int))
  copied = original
  tl.source_integer copied = tl.int64
  offsets = tl.arange(0, 32)
  random = tl.rand(copied, offsets)
}

-- Unqualified lookalikes cannot impersonate the proof-checked rooted adapter.
private def i32ArangeSize (n : Nat) : Nat := n

/-- error: tl.rand: offsets have no supported source integer profile; unknown symbolic sizes need a checked rooted i32ArangeSize adapter -/
#guard_msgs in
def fakeAdapter (seed : Int) (B : Nat) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, $(i32ArangeSize B))
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.rand: seed has no supported source integer profile; annotate a direct external scalar binding with tl.source_integer -/
#guard_msgs in
def constexprArithmeticNeedsOwnBound (a b : Nat) (ha : a < 2^31) (hb : b < 2^31) :
    ComputeKernel := triton {
  a32 = $(_root_.VeriTile.Triton.Random.i32NatConstexpr a ha)
  b32 = $(_root_.VeriTile.Triton.Random.i32NatConstexpr b hb)
  seed_arg = a32 + b32
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.rand: seed has no supported source integer profile; annotate a direct external scalar binding with tl.source_integer -/
#guard_msgs in
def branchDifferentProfile (left right : Int) (choose : Bool) : ComputeKernel := triton {
  left_seed = $((left : Int))
  tl.source_integer left_seed = tl.int32
  right_seed = $((right : Int))
  tl.source_integer right_seed = tl.int64
  if $((choose : Bool)) {
    seed_arg = left_seed
  } else {
    seed_arg = right_seed
  }
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.rand: seed has no supported source integer profile; annotate a direct external scalar binding with tl.source_integer -/
#guard_msgs in
def branchReassignClears (seed next : Int) (choose : Bool) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  if $((choose : Bool)) {
    seed_arg = $((next : Int))
  }
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets)
}

/-- error: tl.__rand_profiled is internal syntax; use tl.rand with source profiles -/
#guard_msgs in
def directInternalProfile (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  offsets = tl.arange(0, 32)
  random = tl.__rand_profiled((⟨.i32, .i32, 10⟩ : Random.SourceProfile), seed_arg, offsets)
}

/-- error: tl.rand: n_rounds must be a known nonnegative constexpr; use the rooted roundsConstexpr adapter for a symbolic source constexpr -/
#guard_msgs in
def runtimeRounds (seed : Int) (rounds : Nat) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  runtime_rounds = $((rounds : Nat))
  offsets = tl.arange(0, 32)
  random = tl.rand(seed_arg, offsets, runtime_rounds)
}

/-- error: tl.rand: offsets have no supported source integer profile; unknown symbolic sizes need a checked rooted i32ArangeSize adapter -/
#guard_msgs in
def invalidRange (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int32
  offsets = tl.arange(0, 33)
  random = tl.rand(seed_arg, offsets)
}

-- Same-shaped vector seeds now use the separate per-lane canonical primitive.
def vectorSeed : ComputeKernel := triton {
  offsets = tl.arange(0, 32)
  random = tl.rand(offsets, offsets)
}

theorem vectorSeed_shape : finalShape? vectorSeed = some [32] := rfl

-- Existing signatures remain usable without new fields or parameters.
example : VeriTile.Triton.DSL.Env =
    List (String × VeriTile.Triton.DSL.DInfo × VeriTile.Triton.DSL.SInfo ×
      Option VeriTile.Triton.DSL.DInfo) := rfl

#print axioms seedI32_profile
#print axioms seedU64_profile
#print axioms zeroRounds_profile
#print axioms symbolicBlock_shape
#print axioms branchSame_profile
#print axioms signed32Loads_profile
#print axioms unsignedLoads_inputs
#print axioms promoted_retainsFP32

end VeriTile.Bench.Tests.RandomFrontend
