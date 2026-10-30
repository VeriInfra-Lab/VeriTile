import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Triton.Semantics.Step

/-! Direct Philox frontend regressions. Tuple targets intentionally overlap
both the seed and counters. The independent summary checks that every input
is captured at its original shape before any result register is changed.
-/
namespace VeriTile.Bench.Tests.DirectRandomFrontend
open VeriTile.Triton

def aliasing (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int64
  c0 = tl.arange(0, 2).to(tl.uint32)
  c1 = (0).to(tl.uint32)
  c2 = c0
  c3 = c0
  seed_arg, c0, c1, c2 = tl.philox(seed_arg, c0, c1, c2, c3)
}

def tupleBody (kernel : ComputeKernel) : List ComputeStmt :=
  match kernel.surfaceBody.getLast? with
  | some (.ifThen _ body) => body
  | _ => []

def assignmentSummary : ComputeStmt → Option (TileDType × TileShape × RegName)
  | .assign dtype shape name _ => some (dtype, shape, name)
  | _ => none

theorem all_inputs_captured_before_aliasing_outputs (seed : Int) :
    (tupleBody (aliasing seed)).map assignmentSummary =
      [some (.int, [], "__veritile_direct_random_0"),
       some (.int, [2], "__veritile_direct_random_1"),
       some (.int, [], "__veritile_direct_random_2"),
       some (.int, [2], "__veritile_direct_random_3"),
       some (.int, [2], "__veritile_direct_random_4"),
       some (.nat, [2], "seed_arg"), some (.nat, [2], "c0"),
       some (.nat, [2], "c1"), some (.nat, [2], "c2")] := rfl

def integerRefName {shape : TileShape} : Op .int shape → Option RegName
  | .ref _ _ name => some name
  | _ => none

def capturedPhilox {dtype : TileDType} {shape : TileShape} : Op dtype shape → Bool
  | .philox4x32 (rounds := rounds) cfg _ _ seed c0 c1 c2 c3 =>
      rounds == 10 && cfg == ⟨.i64, .u32, .u32, .u32, .u32⟩ &&
      [integerRefName seed, integerRefName c0, integerRefName c1,
        integerRefName c2, integerRefName c3] ==
      [some "__veritile_direct_random_0", some "__veritile_direct_random_1",
       some "__veritile_direct_random_2", some "__veritile_direct_random_3",
       some "__veritile_direct_random_4"]
  | _ => Bool.false

def capturedAssignment : ComputeStmt → Bool
  | .assign _ _ _ (.alg operation) => capturedPhilox operation
  | _ => Bool.false

theorem every_field_reads_all_original_captured_inputs (seed : Int) :
    ((tupleBody (aliasing seed)).drop 5).map capturedAssignment =
      [Bool.true, Bool.true, Bool.true, Bool.true] := rfl

def nestedLoads (Seeds C0 C1 C2 C3 : Region .int) : ComputeKernel := triton {
  tl.region Seeds = tl.int64, C0 = tl.int32, C1 = tl.int32, C2 = tl.int32, C3 = tl.int32
  r0, r1, r2, r3 = tl.philox(tl.load(Seeds), tl.load(C0), tl.load(C1), tl.load(C2), tl.load(C3))
}

theorem tuple_metadata_preserves_all_five_load_inputs (Seeds C0 C1 C2 C3 : Region .int) :
    (nestedLoads Seeds C0 C1 C2 C3).inputs = [Seeds, C0, C1, C2, C3] := rfl

/-- error: tl.philox: seed requires an explicit supported source integer profile -/
#guard_msgs in
def unknownSeed (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  r0, r1, r2, r3 = tl.philox(seed_arg, 0, 0, 0, 0)
}

/-- error: tl.philox: every counter requires an int32 or uint32 source profile -/
#guard_msgs in
def counter64 (counter : Int) : ComputeKernel := triton {
  counter_arg = $((counter : Int))
  tl.source_integer counter_arg = tl.int64
  r0, r1, r2, r3 = tl.philox(0, counter_arg, 0, 0, 0)
}

/-- error: source identifiers cannot use the reserved __veritile_direct_random_ prefix -/
#guard_msgs in
def captureCollision : ComputeKernel := triton {
  __veritile_direct_random_0 = 1
  r0, r1, r2, r3 = tl.philox(0, 0, 0, 0, 0)
}

/-- error: tl.__philox_profiled is internal syntax; use tl.philox -/
#guard_msgs in
def forgedProfile : ComputeKernel := triton {
  r0, r1, r2, r3 = tl.__philox_profiled(
    (⟨.i32, .i32, .i32, .i32, .i32⟩ : Random.Direct.Profile), 0, 0, 0, 0, 0)
}

def uniformAlias (word : Nat) : ComputeKernel := triton {
  tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float
  word_arg = $((word : Nat))
  tl.source_integer word_arg = tl.uint32
  uniform = uint_to_uniform_float(word_arg)
}

theorem alias_has_fixed_fp32_conversion (word : Nat) :
    (uniformAlias word).surfaceBody.getLast? = some
      (.assign .real [] "uniform" (.compute (.alg .fp32
        (.uniform32 (.ref .nat [] "word_arg"))))) := rfl

def mixedAdd (word : Nat) (increment : Int) : ComputeKernel := triton {
  word_arg = $((word : Nat))
  tl.source_integer word_arg = tl.uint32
  increment_arg = $((increment : Int))
  tl.source_integer increment_arg = tl.int32
  word_arg += increment_arg
}

theorem mixed_add_has_actual_modulo (word : Nat) (increment : Int) :
    (mixedAdd word increment).surfaceBody.getLast? = some
      (.assign .nat [] "word_arg" (.alg
        (Op.intToUnsigned 32 (.add .int .nil
          (.castNatToInt (.ref .nat [] "word_arg")) (.ref .int [] "increment_arg"))))) := rfl

noncomputable def mixedResult (word : Nat) (increment : Int) (state : BlockState) :=
  ((mixedAdd word increment).toAlgorithm?.toOption.bind
    (fun kernel => VeriTile.Triton.exec kernel state)).bind
      (fun result => result.regs .nat [] "word_arg")

theorem mixed_add_exec (word : Nat) (increment : Int) (state : BlockState) :
    mixedResult word increment state =
      some (Tile.scalar ((((word : Int) + increment) % 2^32).toNat)) := by
  simp only [mixedResult, mixedAdd, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_intToUnsigned]
  simp [evalOp, BlockState.setReg, Tile.scalar, Tile.bop, NumericDType.add,
    Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta]

theorem mixed_add_overflow (state : BlockState) :
    mixedResult 4294967295 2 state = some (Tile.scalar 1) := mixed_add_exec _ _ _

theorem mixed_add_negative (state : BlockState) :
    mixedResult 0 (-1) state = some (Tile.scalar 4294967295) := mixed_add_exec _ _ _

def signedCounters (offset : Int) : ComputeKernel := triton {
  offset_arg = $((offset : Int))
  tl.source_integer offset_arg = tl.int64
  low = (offset_arg & 0xFFFFFFFF).to(tl.uint32)
  high = ((offset_arg >> 32) & 0xFFFFFFFF).to(tl.uint32)
}

theorem signed_counter_lowering (offset : Int) :
    (signedCounters offset).surfaceBody =
      [.assign .int [] "offset_arg" (.alg (.constInt offset)),
       .assign .nat [] "low" (.alg (Op.intToUnsigned 32
         (Op.intBitAnd 64 .nil (.ref .int [] "offset_arg") (.constInt 4294967295)))),
       .assign .nat [] "high" (.alg (Op.intToUnsigned 32
         (Op.intBitAnd 64 .nil
           (Op.intShiftRight .nil (.ref .int [] "offset_arg") (.constNat 32))
           (.constInt 4294967295))))] := rfl

noncomputable def counterResults (offset : Int) (state : BlockState) :=
  ((signedCounters offset).toAlgorithm?.toOption.bind
    (fun kernel => VeriTile.Triton.exec kernel state)).map
      (fun result => (result.regs .nat [] "low", result.regs .nat [] "high"))

theorem negative_offset_exec (state : BlockState) :
    counterResults (-1) state = some
      (some (Tile.scalar 4294967295), some (Tile.scalar 4294967295)) := by
  simp only [counterResults, signedCounters, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_intToUnsigned, eval_intBitAnd, eval_intShiftRight]
  simp [evalOp, BlockState.setReg, Tile.scalar, -Tile.scalar_eta]
  rfl

def uniformSourceShape (seed offset : Int) (BLOCK : Nat)
    (hBlock : Random.ArangeSizeDomain BLOCK) : ComputeKernel := triton {
  tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float
  philox_seed = $((seed : Int))
  tl.source_integer philox_seed = tl.int64
  philox_offset = $((offset : Int))
  tl.source_integer philox_offset = tl.int64
  BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hBlock)
  philox_seed = philox_seed.to(tl.int64)
  philox_offset = philox_offset.to(tl.int64)
  c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
  c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
  i4 = tl.program_id(0) * BLOCK_arg + tl.arange(0, BLOCK_arg)
  c0 += i4
  _O = c0 * 0
  r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
  r0 = uint_to_uniform_float(r0)
  r1 = uint_to_uniform_float(r1)
  r2 = uint_to_uniform_float(r2)
  r3 = uint_to_uniform_float(r3)
}

theorem actual_source_mixed_shapes (seed offset : Int) (BLOCK : Nat)
    (hBlock : Random.ArangeSizeDomain BLOCK) :
    (((uniformSourceShape seed offset BLOCK hBlock).surfaceBody.reverse.take 4).reverse).map assignmentSummary =
      [some (.real, [BLOCK], "r0"), some (.real, [BLOCK], "r1"),
       some (.real, [BLOCK], "r2"), some (.real, [BLOCK], "r3")] := rfl

/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
def unboundHelper : ComputeKernel := triton {
  random = uint_to_uniform_float((0).to(tl.uint32))
}

/-- error: cannot rebind an explicitly declared random source helper -/
#guard_msgs in
def reboundHelper : ComputeKernel := triton {
  tl.source_helper helper = tl.uint_to_uniform_float
  helper = 0
}

/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
def singleBranchHelper (choose : Bool) : ComputeKernel := triton {
  if $((choose : Bool)) {
    tl.source_helper helper = tl.uint_to_uniform_float
  }
  random = helper((0).to(tl.uint32))
}

/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
def loopShadowedHelper : ComputeKernel := triton {
  tl.source_helper helper = tl.uint_to_uniform_float
  for helper in range(1) {
    unused = 0
  }
  random = helper((0).to(tl.uint32))
}

/-- error: signed int64 right shift requires a literal count below 64 -/
#guard_msgs in
def negativeShift (offset : Int) : ComputeKernel := triton {
  offset_arg = $((offset : Int))
  tl.source_integer offset_arg = tl.int64
  bad = offset_arg >> -1
}

/-- error: signed int64 right shift requires a literal count below 64 -/
#guard_msgs in
def wideShift (offset : Int) : ComputeKernel := triton {
  offset_arg = $((offset : Int))
  tl.source_integer offset_arg = tl.int64
  bad = offset_arg >> 64
}

def storeFirst (out : Region TileDType.real) : ComputeKernel := triton {
  tl.store(out, 0.0, mask=0 == 0, eviction_policy="evict_first")
}

def storeLast (out : Region TileDType.real) : ComputeKernel := triton {
  tl.store(out, 0.0, mask=0 == 0, eviction_policy="evict_last")
}

def storeDefault (out : Region TileDType.real) : ComputeKernel := triton {
  tl.store(out, 0.0, mask=0 == 0)
}

theorem store_cache_hints_preserve_memory_semantics (out : Region TileDType.real) :
    storeFirst out = storeDefault out ∧ storeLast out = storeDefault out := ⟨rfl, rfl⟩

-- The literal-only keyword rejects dynamic hints during parsing, before
-- elaboration-level #guard_msgs can intercept the diagnostic.
run_cmd do
  let parserEnv ← Lean.getEnv
  match Lean.Parser.runParserCategory parserEnv `tritonStmt
      "tl.store(out, 0.0, mask=0 == 0, eviction_policy=$(policy))" with
  | .error _ => pure ()
  | .ok _ => Lean.throwError "dynamic store cache hint unexpectedly parsed"

/-- error: tl.store: eviction_policy requires literal evict_first or evict_last -/
#guard_msgs in
def unknownStoreHint (out : Region TileDType.real) : ComputeKernel := triton {
  tl.store(out, 0.0, mask=0 == 0, eviction_policy="unknown")
}

/-- error: tl.store: duplicate `eviction_policy=` kwarg -/
#guard_msgs in
def duplicateStoreHint (out : Region TileDType.real) : ComputeKernel := triton {
  tl.store(out, 0.0, mask=0 == 0, eviction_policy="evict_first", eviction_policy="evict_last")
}

#axiomsClean all_inputs_captured_before_aliasing_outputs
#axiomsClean every_field_reads_all_original_captured_inputs
#axiomsClean tuple_metadata_preserves_all_five_load_inputs
#axiomsClean alias_has_fixed_fp32_conversion
#axiomsClean mixed_add_has_actual_modulo
#axiomsClean mixed_add_exec
#axiomsClean mixed_add_overflow
#axiomsClean mixed_add_negative
#axiomsClean signed_counter_lowering
#axiomsClean negative_offset_exec
#axiomsClean actual_source_mixed_shapes
#axiomsClean store_cache_hints_preserve_memory_semantics

end VeriTile.Bench.Tests.DirectRandomFrontend
