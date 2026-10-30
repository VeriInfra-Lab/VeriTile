import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

/-! Bit-level building block for the two pending packed-FP4 imports. This is
not a source execution contract: interleave, runtime bitcast and the complete
masked kernel still need a translation/execution proof. The pinned source uses
the high nibble first, despite the names of its temporary variables. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.Bits
open VeriTile.Triton
abbrev Nibble := BitVec 4
abbrev Packed := BitVec 8

/-- Source helper's fixed e2m1 constants and fixed-width operations. The first
integer stage is uint8; the shifts that build the float payload are uint32.
The two exceptional magnitudes are selected before attaching the sign bit. -/
def sourceWord32 (n : Nibble) : BitVec 32 :=
  let x : BitVec 8 := n.zeroExtend 8
  let sign := x &&& (8 : BitVec 8)
  let positive := x ^^^ sign
  let exponent := ((positive >>> 1) - (1 : BitVec 8) + (127 : BitVec 8)).zeroExtend 32 <<< 23
  let mantissa := (positive &&& (1 : BitVec 8)).zeroExtend 32 <<< 22
  let normal := exponent ||| mantissa
  let zeroHandled := if positive = 0 then (0 : BitVec 32) else normal
  let denormalHandled := if positive = 1 then (0x3f000000 : BitVec 32) else zeroHandled
  denormalHandled ||| (sign.zeroExtend 32 <<< 28)

/-- Independent format table, not defined using the source bit manipulations. -/
def magnitude (n : Nibble) : Rat :=
  match n.toNat % 8 with
  | 0 => 0 | 1 => 1/2 | 2 => 1 | 3 => 3/2
  | 4 => 2 | 5 => 3 | 6 => 4 | _ => 6

def expectedBits (n : Nibble) : BitVec 16 :=
  let positive : Nat := match n.toNat % 8 with
    | 0 => 0 | 1 => 0x3f00 | 2 => 0x3f80 | 3 => 0x3fc0
    | 4 => 0x4000 | 5 => 0x4040 | 6 => 0x4080 | _ => 0x40c0
  BitVec.ofNat 16 (positive + if n.toNat < 8 then 0 else 0x8000)

/-- Direct ordinary fp32-to-bf16 conversion, retaining the sign of zero. -/
def narrow (x : FP.Value .fp32) : FP.Value .bf16 :=
  match x.decode with
  | .nan .. => FP.Value.qNaN .bf16
  | .infinity sign => FP.Value.infinity .bf16 sign
  | .finite sign q => FP.round .bf16 .ieeeValue (if sign then -q else q) sign

def decoded (n : Nibble) : FP.Value .bf16 := narrow ⟨sourceWord32 n⟩

/-- Exhaustive kernel-checked proof for all 16 values, including negative zero.
The conversion is exact in both formats; no numerical hypothesis is assumed. -/
theorem all_nibbles : ∀ n : Nibble,
    (⟨sourceWord32 n⟩ : FP.Value .fp32).decode = .finite (decide (8 ≤ n.toNat)) (magnitude n) ∧
    (decoded n).bits = expectedBits n ∧
    (decoded n).decode = .finite (decide (8 ≤ n.toNat)) (magnitude n) := by
  decide +kernel

def high (x : Packed) : Nibble := (x >>> 4).truncate 4
def low (x : Packed) : Nibble := x.truncate 4

/-- High nibble then low nibble, exactly the order supplied to interleave. -/
def decodedPair (x : Packed) : FP.Value .bf16 × FP.Value .bf16 :=
  (decoded (high x), decoded (low x))

theorem every_packed_byte (x : Packed) :
    (decodedPair x).1.bits = expectedBits (high x) ∧
    (decodedPair x).2.bits = expectedBits (low x) :=
  ⟨(all_nibbles (high x)).2.1, (all_nibbles (low x)).2.1⟩

/-- Distinguishes high-first packing and the two zero signs. -/
theorem ordering_and_zero_vectors :
    (decodedPair 0x21).1.bits = 0x3f80 ∧ (decodedPair 0x21).2.bits = 0x3f00 ∧
    (decodedPair 0x80).1.bits = 0x8000 ∧ (decodedPair 0x80).2.bits = 0 := by
  decide +kernel

#axiomsClean all_nibbles
#axiomsClean every_packed_byte
#axiomsClean ordering_and_zero_vectors
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.Bits
