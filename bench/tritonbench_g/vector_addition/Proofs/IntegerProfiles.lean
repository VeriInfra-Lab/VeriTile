import bench.tritonbench_g.vector_addition.Proofs.IntegerSource

namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerSource
open VeriTile.Triton FP.Structural
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

inductive Canonical where
  | b1 | i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64
  deriving DecidableEq, Repr

def Canonical.storage : Canonical → Storage
  | .b1 => .boolean
  | .i8 => .signed 8 | .u8 => .unsigned 8
  | .i16 => .signed 16 | .u16 => .unsigned 16
  | .i32 => .signed 32 | .u32 => .unsigned 32
  | .i64 => .signed 64 | .u64 => .unsigned 64

def Canonical.dtype : Canonical → DTypeScope.DType
  | .b1 => .b1 | .i8 => .i8 | .u8 => .u8 | .i16 => .i16 | .u16 => .u16
  | .i32 => .i32 | .u32 => .u32 | .i64 => .i64 | .u64 => .u64

def observedFormat (d : DTypeScope.DType) : Format := ⟨d.width,d.signed⟩

/-- All 81 integer/bool promotion-width/signedness branches match the frozen
frontend dispatch rule. This is not a native compilation or error theorem. -/
theorem canonical_promotion (x y : Canonical) :
    (DTypeScope.promote x.dtype y.dtype).map observedFormat =
      some (promote x.storage y.storage) := by
  cases x <;> cases y <;> decide

/-- Every output conversion uses the source pointer dtype; bool is i8. -/
theorem canonical_store (x : Canonical) :
    observedFormat (DTypeScope.storeElement x.dtype) = x.storage.storeFormat := by
  cases x <;> decide

/-- Regression distinction: i1 addition wraps before the bool i8 store.
This is a concrete bitvector statement, not an assumed arithmetic law. -/
theorem bool_true_plus_true_wraps :
    stored (α := Unit) .boolean
      (castValue (promote .boolean .boolean)
        (castValue (promote .boolean .boolean) 1 +
         castValue (promote .boolean .boolean) 1)) = .mk .int 0 := by
  unfold stored
  congr 1

theorem int8_overflow_wraps :
    stored (α := Unit) (.signed 8)
      (castValue (promote (.signed 8) (.signed 8)) (127+1)) = .mk .int (-128) := by
  unfold stored
  congr 1

/-- The i8 output for bool + integer preserves the converted byte; it is not
normalized to one merely because that byte is nonzero. -/
theorem bool_int8_store_is_not_nonzero_normalization :
    stored (α := Unit) .boolean
      (castValue (promote .boolean (.signed 8)) (1+2)) = .mk .int 3 := by
  unfold stored
  congr 1

#axiomsClean canonical_promotion
#axiomsClean canonical_store
#axiomsClean bool_true_plus_true_wraps
#axiomsClean int8_overflow_wraps
#axiomsClean bool_int8_store_is_not_nonzero_normalization
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerSource
