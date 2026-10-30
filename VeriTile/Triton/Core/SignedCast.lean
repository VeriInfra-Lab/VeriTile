import VeriTile.Triton.Core.FixedInteger

/-! Signed narrowing is a derived integer expression. It keeps the low bits,
then interprets their sign; it does not clamp negative values to Nat. -/
namespace VeriTile.Triton

abbrev Op.intToSigned (width : Nat) (value : Op .int shape) : Op .int shape :=
  Op.unsignedToInt width (Op.intToUnsigned width value)

/-- Native Bool-to-signed-integer conversion zero-extends the one-bit value. -/
abbrev Op.boolToInt (value : Op .bool shape) : Op .int shape :=
  .where value (.full shape (.constInt 1)) (.full shape (.constInt 0))

end VeriTile.Triton
