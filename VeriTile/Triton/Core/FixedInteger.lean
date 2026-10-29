import VeriTile.Triton.Core.Integer

/-! Derived fixed-width integer expressions. These retain signed values until
the explicit unsigned conversion. The width and shift-count legality of the
native source remain source-domain obligations. No floating rules are used. -/
namespace VeriTile.Triton
open IntegerExpr

/-- Signed interpretation of the low `width` bits. The doubled comparison also
handles width zero and agrees directly with `BitVec.toInt`. -/
abbrev Op.unsignedToInt (width : Nat) (value : Op .nat shape) : Op .int shape :=
  let bits := Op.natToUnsigned width value
  let signed := Op.castNatToInt bits
  let same := sameShape shape
  .where (.lt .nat same
      (.mul .nat same (.full shape (.constNat 2)) bits)
      (.full shape (.constNat (2^width))))
    signed (.sub .int same signed (.full shape (.constInt (2^width))))

/-- Two's-complement bitwise AND, including negative signed operands. -/
abbrev Op.intBitAnd (width : Nat) (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) : Op .int out :=
  Op.unsignedToInt width
    (.bitAnd bc (Op.intToUnsigned width lhs) (Op.intToUnsigned width rhs))

/-- Arithmetic signed right shift. Division by positive `2^count` uses
Euclidean division, so `-1 >> count` remains `-1`. Native shift-count bounds
must be supplied separately; a signed count must not be clamped to Nat. -/
abbrev Op.intShiftRight (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .nat b) : Op .int out :=
  .floorDiv .int bc lhs
    (.castNatToInt (.shiftLeft (sameShape b) (.full b (.constNat 1)) rhs))

/-- Left shift in the mathematical signed-index projection. Like signed
index multiplication, this does not silently wrap. A native-width refinement
must establish its range, or use an explicit narrowing operation. -/
abbrev Op.intShiftLeft (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .nat b) : Op .int out :=
  .mul .int bc lhs
    (.castNatToInt (.shiftLeft (sameShape b) (.full b (.constNat 1)) rhs))

end VeriTile.Triton
