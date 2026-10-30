import VeriTile.Triton.DSL
import VeriTile.Triton.Float.UnsignedCast
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! Unsigned seed casts preserve low bits, including negative int64 inputs.
These are execution tests of the DSL expansion, not just AST-shape assertions. -/
namespace VeriTile.Bench.Tests.UnsignedIntegerCast
open VeriTile.Triton

def signed (x : Int) : ComputeKernel := triton {
  value = $((x : Int))
  u32 = value.to(tl.uint32)
  u64 = (value).to(tl.uint64)
  functional = tl.cast(value, tl.uint32)
}

noncomputable def results (k : ComputeKernel) (s : BlockState) :=
  (k.toAlgorithm?.toOption.bind (fun a => VeriTile.Triton.exec a s)).map
    (fun t => (t.regs .nat [] "u32", t.regs .nat [] "u64", t.regs .nat [] "functional"))

theorem signed_exec (x : Int) (s : BlockState) :
    results (signed x) s = some
      (some (Tile.scalar ((x % (2^32 : Int)).toNat)),
       some (Tile.scalar ((x % (2^64 : Int)).toNat)),
       some (Tile.scalar ((x % (2^32 : Int)).toNat))) := by
  simp only [results, signed, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_intToUnsigned]
  simp [evalOp, BlockState.setReg, Tile.scalar, -Tile.scalar_eta]

theorem negative_seed (s : BlockState) :
    results (signed (-1)) s = some
      (some (Tile.scalar 4294967295), some (Tile.scalar 18446744073709551615),
       some (Tile.scalar 4294967295)) := signed_exec (-1) s

theorem high_bits (s : BlockState) :
    results (signed (2^32 + 7)) s = some
      (some (Tile.scalar 7), some (Tile.scalar 4294967303),
       some (Tile.scalar 7)) := signed_exec (2^32 + 7) s

def unsigned (x : Nat) : ComputeKernel := triton {
  value = $((x : Nat))
  narrowed = value.to(tl.uint32)
}

theorem unsigned_lowering (x : Nat) : (unsigned x).surfaceBody = [
    .assign .nat [] "value" (.alg (.constNat x)),
    .assign .nat [] "narrowed" (.alg (Op.natToUnsigned 32 (.ref .nat [] "value")))] := rfl

def loadedSeed (Seeds : RegionName) : ComputeKernel := triton {
  tl.region Seeds = tl.int64
  seed = tl.load(Seeds + tl.program_id(0)).to(tl.uint32)
  random = tl.rand(seed, tl.arange(0, 4), n_rounds=7)
}

theorem loaded_seed_storage (Seeds : RegionName) :
    (loadedSeed Seeds).surfaceBody.head? = some
      (.assign .nat [] "seed" (.alg (Op.intToUnsigned 32
        (.load .int (.region Seeds (.programId 0)) .none .zero)))) := rfl

theorem loaded_seed_profile (Seeds : RegionName) :
    ((loadedSeed Seeds).surfaceBody.reverse.head?).map
      (fun stmt => match stmt with
        | .assign _ _ _ (.compute (.alg .fp32 (.rand cfg _ _))) => some cfg
        | _ => none) = some (some ⟨.u32, .i32, 7⟩) := rfl

/-- Without a region declaration, direct unsigned load syntax keeps its
existing Nat storage channel and still applies the requested modulo width. -/
def unknownRootLoads (Seeds : RegionName) : ComputeKernel := triton {
  u32 = tl.load(Seeds + tl.program_id(0)).to(tl.uint32)
  u64 = tl.load(Seeds + tl.program_id(0)).to(tl.uint64)
}

theorem unknown_root_direct_loads (Seeds : RegionName) :
    (unknownRootLoads Seeds).surfaceBody =
      [.assign .nat [] "u32" (.alg (Op.natToUnsigned 32
        (.load .nat (.region Seeds (.programId 0)) .none .zero))),
       .assign .nat [] "u64" (.alg (Op.natToUnsigned 64
        (.load .nat (.region Seeds (.programId 0)) .none .zero)))] := rfl

/-- Nested loads must keep their signed source channel as well. Splitting a
seed into a separate assignment would not test the original layer-norm form. -/
def nestedLoadedSeeds (Seeds : RegionName) : ComputeKernel := triton {
  tl.region Seeds = tl.int64
  direct = tl.rand(tl.load(Seeds + tl.program_id(0)).to(tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  parenthesized = tl.rand((tl.load(Seeds + tl.program_id(0))).to(tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  functional = tl.rand(tl.cast(tl.load(Seeds + tl.program_id(0)), tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  deep_parens = tl.rand((((tl.load(Seeds + tl.program_id(0))))).to(tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  functional_parens = tl.rand(tl.cast(((tl.load(Seeds + tl.program_id(0)))), tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  explicit_dtype = tl.rand(tl.load(Seeds + tl.program_id(0), dtype=tl.int64).to(tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  explicit_parens = tl.rand(((tl.load(Seeds + tl.program_id(0), dtype=tl.int64))).to(tl.uint32),
    tl.arange(0, 4), n_rounds=7)
  explicit_functional = tl.rand(tl.cast((tl.load(Seeds + tl.program_id(0), dtype=tl.int64)), tl.uint32),
    tl.arange(0, 4), n_rounds=7)
}

def nestedRandAssignment (Seeds : RegionName) (name : RegName) : ComputeStmt :=
  .assign .real [4] name (.compute (.alg .fp32
    (.rand ⟨.u32, .i32, 7⟩
      (.castNatToInt (Op.intToUnsigned 32
        (.load .int (.region Seeds (.programId 0)) .none .zero)))
      (.castNatToInt (.arange 4)))))

theorem nested_signed_seed_storage (Seeds : RegionName) :
    (nestedLoadedSeeds Seeds).surfaceBody =
      [nestedRandAssignment Seeds "direct", nestedRandAssignment Seeds "parenthesized",
       nestedRandAssignment Seeds "functional", nestedRandAssignment Seeds "deep_parens",
       nestedRandAssignment Seeds "functional_parens", nestedRandAssignment Seeds "explicit_dtype",
       nestedRandAssignment Seeds "explicit_parens", nestedRandAssignment Seeds "explicit_functional"] := rfl

/-- The uint64 destination has its own modulo width and source profile. -/
def nestedLoadedSeeds64 (Seeds : RegionName) : ComputeKernel := triton {
  tl.region Seeds = tl.int64
  direct64 = tl.rand(tl.load(Seeds + tl.program_id(0)).to(tl.uint64), tl.arange(0, 4), 7)
  deep64 = tl.rand((((tl.load(Seeds + tl.program_id(0))))).to(tl.uint64), tl.arange(0, 4), 7)
  functional64 = tl.rand(tl.cast(((tl.load(Seeds + tl.program_id(0), dtype=tl.int64))), tl.uint64),
    tl.arange(0, 4), 7)
}

def nestedRandAssignment64 (Seeds : RegionName) (name : RegName) : ComputeStmt :=
  .assign .real [4] name (.compute (.alg .fp32
    (.rand ⟨.u64, .i32, 7⟩
      (.castNatToInt (Op.intToUnsigned 64
        (.load .int (.region Seeds (.programId 0)) .none .zero)))
      (.castNatToInt (.arange 4)))))

theorem nested_signed_seed64_storage (Seeds : RegionName) :
    (nestedLoadedSeeds64 Seeds).surfaceBody =
      [nestedRandAssignment64 Seeds "direct64", nestedRandAssignment64 Seeds "deep64",
       nestedRandAssignment64 Seeds "functional64"] := rfl

/-- Direct stores must receive the same load/cast annotation as assignments. -/
def directlyStoredRandom (Seeds Out : RegionName) : ComputeKernel := triton {
  tl.region Seeds = tl.int64
  tl.store(Out + tl.arange(0, 4),
    tl.rand(tl.load(Seeds + tl.program_id(0), dtype=tl.int64).to(tl.uint32),
      tl.arange(0, 4), n_rounds=7))
}

theorem direct_store_retains_signed_seed (Seeds Out : RegionName) :
    (directlyStoredRandom Seeds Out).surfaceBody =
      [.store .real [4] (.region Out (.arange 4)) (.compute (.alg .fp32
        (.rand ⟨.u32, .i32, 7⟩
          (.castNatToInt (Op.intToUnsigned 32
            (.load .int (.region Seeds (.programId 0)) .none .zero)))
          (.castNatToInt (.arange 4))))) .none] := rfl

/-- Conditions are a separate expression entry point from assignment RHSs. -/
def unsignedLoadCondition (Seeds : RegionName) : ComputeKernel := triton {
  tl.region Seeds = tl.int64
  if tl.load(Seeds + tl.program_id(0)).to(tl.uint32) != $((0 : Nat)) {
    selected = $((1 : Nat))
  }
}

theorem condition_retains_signed_load (Seeds : RegionName) :
    (unsignedLoadCondition Seeds).surfaceBody =
      [.ifThen (.ne .nat .nil
        (Op.intToUnsigned 32 (.load .int (.region Seeds (.programId 0)) .none .zero))
        (.constNat 0)) [.assign .nat [] "selected" (.alg (.constNat 1))]] := rfl

/-- Only the selected signed memory cell contains the seed; all other cells
are Nat zero. Reading the seed through the wrong channel would produce zero. -/
def signedSeedState (Seeds : RegionName) (seed : Int) : BlockState :=
  { mem := fun region offset =>
      if region = Seeds ∧ offset = 3 then MemCell.of .int seed else MemCell.of .nat 0
    regs := fun _ _ _ => none
    pids := fun _ => 3 }

private theorem eval_signed_seed_load (Seeds : RegionName) (seed : Int) :
    evalOp (.load .int (.region Seeds (.programId 0)) .none .zero) (signedSeedState Seeds seed) =
      some (Tile.scalar seed) := by
  have hmem : (signedSeedState Seeds seed).mem Seeds 3 = MemCell.of .int seed := by
    simp [signedSeedState]
  have hread : (signedSeedState Seeds seed).readMemValue .int Seeds 3 = seed := by
    simp only [BlockState.readMemValue, BlockState.readMemTyped, hmem, MemCell.readAs_of_same]
  simp only [evalOp, bind, Option.bind_some, ↓reduceIte]
  change some (⟨fun _ => (signedSeedState Seeds seed).readMemValue .int Seeds 3⟩ : Tile .int []) = _
  rw [hread]
  rfl

private theorem eval_loaded_unsigned_seed (Seeds : RegionName) (seed : Int) (width : Nat) :
    evalOp (.castNatToInt (Op.intToUnsigned width
      (.load .int (.region Seeds (.programId 0)) .none .zero))) (signedSeedState Seeds seed) =
      some (Tile.scalar (Int.ofNat ((seed % (2^width : Int)).toNat))) := by
  have cast_eval (value : Op .nat []) :
      evalOp (.castNatToInt value) (signedSeedState Seeds seed) = (do
        let v ← evalOp value (signedSeedState Seeds seed)
        pure (⟨fun i => Int.ofNat (v.data i)⟩ : Tile .int [])) := by
    simp only [evalOp, pure]
  rw [cast_eval, eval_intToUnsigned, eval_signed_seed_load]
  rfl

private def randomSeed? : ComputeStmt → Option (Op .int [])
  | .assign _ _ _ (.compute (.alg .fp32 (.rand _ seed _))) => some seed
  | _ => none

private noncomputable def actualSeedValue (state : BlockState) (stmt : ComputeStmt) : Option Int :=
  ((randomSeed? stmt).bind (fun seed => evalOp seed state)).map (fun tile => tile.data PUnit.unit)

private theorem actual_seed_value_of_extracted (state : BlockState) (stmt : ComputeStmt)
    (seedOp : Op .int []) (value : Int)
    (hextracted : randomSeed? stmt = some seedOp)
    (hevaluated : evalOp seedOp state = some (Tile.scalar value)) :
    actualSeedValue state stmt = some value := by
  unfold actualSeedValue
  rw [hextracted]
  simp only [Option.bind_some]
  rw [hevaluated]
  rfl

private theorem actual_seed_value32 (Seeds : RegionName) (seed : Int) (name : RegName) :
    actualSeedValue (signedSeedState Seeds seed) (nestedRandAssignment Seeds name) =
      some (Int.ofNat ((seed % (2^32 : Int)).toNat)) := by
  exact actual_seed_value_of_extracted _ _ _ _ rfl
    (eval_loaded_unsigned_seed Seeds seed 32)

private theorem actual_seed_value64 (Seeds : RegionName) (seed : Int) (name : RegName) :
    actualSeedValue (signedSeedState Seeds seed) (nestedRandAssignment64 Seeds name) =
      some (Int.ofNat ((seed % (2^64 : Int)).toNat)) := by
  exact actual_seed_value_of_extracted _ _ _ _ rfl
    (eval_loaded_unsigned_seed Seeds seed 64)

/-- Evaluate the seed extracted from every actual expanded syntax variant,
including explicit-dtype and multiply parenthesized nested casts. -/
theorem negative_memory_seed_values (Seeds : RegionName) :
    (nestedLoadedSeeds Seeds).surfaceBody.map (actualSeedValue (signedSeedState Seeds (-1))) =
      List.replicate 8 (some (4294967295 : Int)) := by
  rw [nested_signed_seed_storage]
  simp only [List.map_cons, List.map_nil, actual_seed_value32, unsigned_cast_minus_one]
  rfl

theorem negative_memory_seed64_values (Seeds : RegionName) :
    (nestedLoadedSeeds64 Seeds).surfaceBody.map (actualSeedValue (signedSeedState Seeds (-1))) =
      List.replicate 3 (some (18446744073709551615 : Int)) := by
  rw [nested_signed_seed64_storage]
  simp only [List.map_cons, List.map_nil, actual_seed_value64]
  rfl

theorem missing_operand (s : BlockState) (h : s.regs .int [4] "absent" = none) :
    evalOp (Op.intToUnsigned 32 (.ref .int [4] "absent")) s = none := by
  simp [eval_intToUnsigned, h]

def randAfterUpdates (seed : Int) (flag : Bool) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int64
  offsets = tl.arange(0, 4)
  value = 2.0
  if $((flag : Bool)) {
    value += 1.0
    value -= 1.0
    value *= 2.0
    value /= 2.0
  }
  random = tl.rand(seed_arg, offsets, n_rounds=7)
}

theorem updates_preserve_unrelated_profile (seed : Int) (flag : Bool) :
    ((randAfterUpdates seed flag).surfaceBody.reverse.head?).map
      (fun stmt => match stmt with
        | .assign _ _ _ (.compute (.alg .fp32 (.rand cfg _ _))) => some cfg
        | _ => none) = some (some ⟨.i64, .i32, 7⟩) := rfl

/-- error: tl.rand: offsets have no supported source integer profile; unknown symbolic sizes need a checked rooted i32ArangeSize adapter -/
#guard_msgs in
def changedOffsets (seed : Int) : ComputeKernel := triton {
  seed_arg = $((seed : Int))
  tl.source_integer seed_arg = tl.int64
  offsets = tl.arange(0, 4)
  offsets += 1
  random = tl.rand(seed_arg, offsets)
}

#print axioms signed_exec
#print axioms negative_seed
#print axioms high_bits
#print axioms loaded_seed_storage
#print axioms loaded_seed_profile
#print axioms FP.Structural.eval_intToUnsigned
#print axioms evalOpR_intToUnsigned
#print axioms unsigned_cast_eq_bitvec
#print axioms updates_preserve_unrelated_profile

#axiomsClean VeriTile.Triton.eval_intToUnsigned
#axiomsClean VeriTile.Triton.evalOpR_intToUnsigned
#axiomsClean VeriTile.Triton.FP.Structural.eval_intToUnsigned
#axiomsClean VeriTile.Triton.eval_natToUnsigned
#axiomsClean VeriTile.Triton.FP.Structural.eval_natToUnsigned
#axiomsClean VeriTile.Triton.unsigned_cast_lt
#axiomsClean VeriTile.Triton.unsigned_cast_of_nonneg_lt
#axiomsClean VeriTile.Triton.unsigned_cast_eq_bitvec
#axiomsClean VeriTile.Triton.unsigned_cast_minus_one
#axiomsClean VeriTile.Triton.unsigned_cast_high_bits
#axiomsClean VeriTile.Triton.unsigned_cast_min_int64
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.signed_exec
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.loaded_seed_storage
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.loaded_seed_profile
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.unknown_root_direct_loads
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.updates_preserve_unrelated_profile
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.nested_signed_seed_storage
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.nested_signed_seed64_storage
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.direct_store_retains_signed_seed
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.condition_retains_signed_load
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.negative_memory_seed_values
#axiomsClean VeriTile.Bench.Tests.UnsignedIntegerCast.negative_memory_seed64_values

end VeriTile.Bench.Tests.UnsignedIntegerCast
