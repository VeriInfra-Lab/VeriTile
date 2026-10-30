import VeriTile.Triton.Random.Philox

/-! Source integer profiles for deterministic Triton random generation.
The mathematical carrier is Int, but the source width and signedness remain
explicit. Values outside their declared source representation are rejected;
the modular generator does not silently validate such source inputs. -/
namespace VeriTile.Triton.Random

inductive IntegerType where
  | i32 | i64 | u32 | u64
  deriving Repr, BEq, DecidableEq, Inhabited

def IntegerType.inRange : IntegerType → Int → Bool
  | .i32, n => decide (-(2^31 : Int) ≤ n ∧ n < 2^31)
  | .i64, n => decide (-(2^63 : Int) ≤ n ∧ n < 2^63)
  | .u32, n => decide (0 ≤ n ∧ n < (2^32 : Int))
  | .u64, n => decide (0 ≤ n ∧ n < (2^64 : Int))

def IntegerType.offsetWidth : IntegerType → OffsetWidth
  | .i32 | .u32 => .bits32
  | .i64 | .u64 => .bits64

/-- A source-selected seed ABI and offset tensor dtype. The default round
count is ten; an explicit zero is retained as zero. There is no default dtype. -/
structure SourceProfile where
  seedType : IntegerType
  offsetType : IntegerType
  nRounds : Nat := 10
  deriving Repr, BEq, DecidableEq

end VeriTile.Triton.Random
