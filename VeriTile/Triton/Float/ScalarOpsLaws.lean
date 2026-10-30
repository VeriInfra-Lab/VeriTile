import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Rat.Lemmas

/-! Exact laws of the concrete scalar software model. These are proved facts,
not empirical admissions. Configuration and format are explicit; no hardware
conformance or general reassociation law follows from them. -/
namespace VeriTile.Triton.FP.ScalarOpsLaws

/-- Addition commutes for every word of either format and every configuration.
Both orders canonicalize NaNs and make the same signed-zero choice, including
exact cancellation under directed rounding. -/
theorem add_comm {f : Format} (cfg : Config) (a b : Value f) :
    FP.add cfg a b = FP.add cfg b a := by
  cases ha : (prepare cfg a).decode <;> cases hb : (prepare cfg b).decode <;>
    simp only [FP.add, ha, hb]
  · rename_i sx x sy y
    change
      round f cfg ((if sx then -x else x) + (if sy then -y else y))
        (if x == 0 && y == 0 && sx == sy then sx else cfg.rounding == .towardNegative) =
      round f cfg ((if sy then -y else y) + (if sx then -x else x))
        (if y == 0 && x == 0 && sy == sx then sy else cfg.rounding == .towardNegative)
    cases sx <;> cases sy <;> simp [_root_.add_comm, Bool.and_comm]
  · rename_i sx sy
    cases sx <;> cases sy <;> rfl

/-- Multiplication commutes for every word of the selected format and every
configuration, including signed zeros, infinities and canonicalized NaNs. -/
theorem mul_comm {f : Format} (cfg : Config) (a b : Value f) :
    FP.mul cfg a b = FP.mul cfg b a := by
  have sign (x y : Bool) : (x != y) = (y != x) := by
    cases x <;> cases y <;> rfl
  cases ha : (prepare cfg a).decode <;> cases hb : (prepare cfg b).decode <;>
    simp [FP.mul, ha, hb, sign, _root_.mul_comm]

/-- Swapping only the two multiplicands of the concrete once-rounded FMA
preserves every word, including signed zeros, infinities and canonical NaNs.
The addend, rounding and flushing configuration remain fixed. -/
theorem fma_mul_comm {f : Format} (cfg : Config) (a b z : Value f) :
    FP.fma cfg a b z = FP.fma cfg b a z := by
  have sign (x y : Bool) : (x != y) = (y != x) := by
    cases x <;> cases y <;> rfl
  simp only [FP.fma, Bool.or_comm (Value.isNaN (prepare cfg a))
    (Value.isNaN (prepare cfg b)),
    Bool.or_comm (Value.isInf (prepare cfg a)) (Value.isInf (prepare cfg b)),
    Bool.and_comm (Value.isInf (prepare cfg b)) (Value.isZero (prepare cfg a)),
    Bool.and_comm (Value.isZero (prepare cfg b)) (Value.isInf (prepare cfg a)),
    Bool.or_comm (Value.isInf (prepare cfg a) && Value.isZero (prepare cfg b))
      (Value.isZero (prepare cfg a) && Value.isInf (prepare cfg b)), sign]
  cases ha : (prepare cfg a).decode <;> cases hb : (prepare cfg b).decode <;>
    cases hz : (prepare cfg z).decode <;> simp [ha, hb, hz, sign, _root_.mul_comm]


/-- The exact positive-one word in either supported format. -/
def one (f : Format) : Value f := Value.ofBits f (127 * 2 ^ f.fractionBits)

theorem one_decode (f : Format) : (one f).decode = .finite Bool.false 1 := by
  cases f <;> decide +kernel

theorem prepare_one {f : Format} (cfg : Config) : prepare cfg (one f) = one f := by
  have normal : (one f).isSubnormal = Bool.false := by cases f <;> decide +kernel
  simp [prepare, normal]

/-- Multiplication by exact positive one agrees with the same-format cast.
The cast explicitly retains input/output flushing and NaN canonicalization;
this is not an unconditional assertion that the original bits are unchanged. -/
theorem mul_one_cast {f : Format} (cfg : Config) (x : Value f) :
    FP.mul cfg x (one f) = FP.cast f cfg x := by
  have hone : (prepare cfg (one f)).decode = .finite Bool.false 1 := by
    rw [prepare_one, one_decode]
  cases hx : (prepare cfg x).decode <;> simp [FP.mul, FP.cast, hx, hone]
  rfl

theorem one_mul_cast {f : Format} (cfg : Config) (x : Value f) :
    FP.mul cfg (one f) x = FP.cast f cfg x :=
  (mul_comm cfg _ _).trans (mul_one_cast cfg x)

/-- Division by exact positive one has the same normalization behavior as
the same-format cast, for every rounding/flush configuration and every word. -/
theorem div_one_cast {f : Format} (cfg : Config) (x : Value f) :
    FP.div cfg x (one f) = FP.cast f cfg x := by
  have hone : (prepare cfg (one f)).decode = .finite Bool.false 1 := by
    rw [prepare_one, one_decode]
  cases hx : (prepare cfg x).decode <;> simp [FP.div, FP.cast, hx, hone]
  rfl

/-- The original closed FP32 profile's positive-zero addition. This does not
claim that adding positive zero preserves every word or every zero sign. -/
theorem add_positive_zero_fp32 :
    FP.add .ieeeValue (Value.zero .fp32) (Value.zero .fp32) = Value.zero .fp32 := by
  decide +kernel

theorem positive_zero_value_fp32 : (Value.zero .fp32).toRat? = some 0 := by
  decide +kernel

theorem abs_positive_zero_fp32 : (Value.zero .fp32).abs = Value.zero .fp32 := rfl

/-- Exact addition commutation with the fixed FP32 value +1, in the original
nearest-even/no-flush profile, including exceptional input words. -/
theorem add_one_comm_fp32 (x : Value .fp32) :
    FP.add .ieeeValue x (Value.ofBits .fp32 0x3f800000) =
      FP.add .ieeeValue (Value.ofBits .fp32 0x3f800000) x :=
  add_comm .ieeeValue x _

#axiomsClean add_comm
#axiomsClean mul_comm
#axiomsClean fma_mul_comm
#axiomsClean one_decode
#axiomsClean prepare_one
#axiomsClean mul_one_cast
#axiomsClean one_mul_cast
#axiomsClean div_one_cast
#axiomsClean add_positive_zero_fp32
#axiomsClean positive_zero_value_fp32
#axiomsClean abs_positive_zero_fp32
#axiomsClean add_one_comm_fp32

end VeriTile.Triton.FP.ScalarOpsLaws
