import VeriTile.Triton
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SignedBitwiseSource
open VeriTile.Triton FP.Structural
set_option maxRecDepth 12000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- A packed native int32 must remain signed through the shift. -/
def source (X : RegionName) (count : Nat) : ComputeKernel := triton {
  tl.region X = tl.int32
  packed = tl.load(X)
  shifted = packed >> $(count)
  nibble = shifted & 15
}

theorem source_keeps_signed_ops (X : RegionName) (count : Nat) :
    (source X count).surfaceBody[1]? = some
      (.assign .int [] "shifted" (.alg (Op.intShiftRight .nil
        (.ref .int [] "packed") (.constNat count)))) ∧
    (source X count).surfaceBody[2]? = some
      (.assign .int [] "nibble" (.alg (Op.intBitAnd 32 .nil
        (.ref .int [] "shifted") (.constInt 15)))) := by
  exact ⟨rfl, rfl⟩

/-- End-to-end negative-word regression. A Nat clamp would produce zero,
whereas arithmetic shift followed by the low-nibble mask produces 15. -/
theorem source_value {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (s : State α) (hword : (s.mem X 0).read .int = -1) :
    ∃ t, FP.Structural.exec M (source X 28) s = some t ∧
      t.regs .int [] "nibble" = some (fun _ => (15 : Int)) := by
  have hload : FP.Structural.evalOp M none
      (Op.load .int (.region (Region.cast (d' := .int) X) (.constNat 0)) .none) s =
      some (fun _ => (s.mem X 0).read .int) := by rfl
  have href (t : State α) (n : RegName) :
      FP.Structural.evalOp M none (Op.ref .int [] n) t = t.regs .int [] n := rfl
  have hnat (t : State α) (n : Nat) :
      FP.Structural.evalOp M none (Op.constNat n) t = some (fun _ => n) := rfl
  have hint (t : State α) (n : Int) :
      FP.Structural.evalOp M none (Op.constInt n) t = some (fun _ => n) := rfl
  simp only [source, ComputeKernel.surfaceBody, FP.Structural.exec, run, step, evalExpr,
    hload, FP.Structural.eval_intShiftRight, FP.Structural.eval_intBitAnd, href, hnat, hint,
    Option.bind_some, pure, bind, State.setReg, ↓reduceIte]
  simp [hword]
  funext i
  exact (by decide : (↑(((-1 : Int) >>> 28 % 4294967296).toNat &&& 15) : Int).bmod 4294967296 = 15)

theorem negative_word_nibble :
    ((BitVec.ofInt 32 ((-1 : Int) >>> 28)) &&& (BitVec.ofNat 32 15)).toInt = 15 := by
  decide

def nested (X Z : RegionName) (count shiftZero : Nat) : ComputeKernel := triton {
  tl.region X = tl.int32, Z = tl.int32
  packed = tl.load(X)
  zero = tl.load(Z)
  value = tl.toReal(((packed >> $(count)) & 15) - ((zero >> $(shiftZero)) & 15))
}

/-- Nested nibble extraction retains signed shifts before its explicit
unsigned low-bit encoding, including under integer-to-float conversion. -/
theorem nested_keeps_signed_ops (X Z : RegionName) (count shiftZero : Nat) :
    (nested X Z count shiftZero).surfaceBody[2]? = some
      (.assign .real [] "value" (.alg (Op.intToReal
        (Op.sub .int .nil
          (Op.intBitAnd 32 .nil
            (Op.intShiftRight .nil (.ref .int [] "packed") (.constNat count)) (.constInt 15))
          (Op.intBitAnd 32 .nil
            (Op.intShiftRight .nil (.ref .int [] "zero") (.constNat shiftZero)) (.constInt 15)))))) := by
  rfl

#axiomsClean source_keeps_signed_ops
#axiomsClean source_value
#axiomsClean negative_word_nibble
#axiomsClean nested_keeps_signed_ops

/-- Later use by tl.toReal must not reclassify a named signed nibble as Nat. -/
def namedConversion (X : RegionName) (count : Nat) : ComputeKernel := triton {
  tl.region X = tl.int32
  packed = tl.load(X)
  nibble = (packed >> $(count)) & 15
  value = tl.toReal(nibble)
}

theorem named_conversion_keeps_signed_ops (X : RegionName) (count : Nat) :
    (namedConversion X count).surfaceBody[1]? = some
      (.assign .int [] "nibble" (.alg (Op.intBitAnd 32 .nil
        (Op.intShiftRight .nil (.ref .int [] "packed") (.constNat count)) (.constInt 15)))) ∧
    (namedConversion X count).surfaceBody[2]? = some
      (.assign .real [] "value" (.alg (Op.intToReal (.ref .int [] "nibble")))) := by
  exact ⟨rfl, rfl⟩

#axiomsClean named_conversion_keeps_signed_ops
end VeriTile.Tests.SignedBitwiseSource
