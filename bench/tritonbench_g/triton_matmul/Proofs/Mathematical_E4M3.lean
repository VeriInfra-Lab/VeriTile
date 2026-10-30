import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.E4M3
open VeriTile.Triton
abbrev Word := FP.Value .fp32
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000

/-- E4M3 finite-only positive encoding: exponent bias 7, three fraction bits.
0x7f is NaN, 0x7e is 448. Half-subnormal step is not reused here. -/
def magnitude (code : Nat) : Rat :=
  let e := code / 8
  let m := code % 8
  if e=0 then (m : Rat)/512 else (8+m : Nat) * FP.pow2 ((e : Int)-10)

def byte (x : Word) : Nat := x.raw % 256

def widen (x : Word) : Word :=
  let b := byte x
  let negative := b≥128
  let c := b%128
  if c=127 then FP.Value.qNaN .fp32
  else if c=0 then FP.Value.zero .fp32 negative
  else FP.round .fp32 .ieeeValue (if negative then -(magnitude c) else magnitude c)

def distance (q : Rat) (c : Nat) := (q-magnitude c).abs

/-- Exhaustive nearest representable positive encoding, ties choose an even
least-significant significand bit. The candidate set explicitly ends at 448,
so overflow saturates. This is a named software policy, not an inferred native
Triton cast rule. -/
def nearest (q : Rat) : Nat :=
  (List.range 127).foldl (fun best c =>
    if distance q c < distance q best ∨ (distance q c = distance q best ∧ c%2=0)
    then c else best) 0

/-- Round-to-nearest-even, finite saturation, signed zero and canonical positive
NaN. These exact conversion choices must be bound to any native claim. -/
def narrow (x : Word) : Word :=
  match x.decode with
  | .nan _ _ _ => FP.Value.ofBits .fp32 127
  | .infinity negative => FP.Value.ofBits .fp32 ((if negative then 128 else 0)+126)
  | .finite negative q => FP.Value.ofBits .fp32 ((if negative then 128 else 0)+nearest q)

/-- Target representation, including minimum subnormal, maximum finite and NaN. -/
theorem decode_boundaries :
    (widen (FP.Value.ofBits .fp32 1)).toRat?=some (1/512 : Rat) ∧
    (widen (FP.Value.ofBits .fp32 56)).toRat?=some 1 ∧
    (widen (FP.Value.ofBits .fp32 126)).toRat?=some 448 ∧
    widen (FP.Value.ofBits .fp32 128)=FP.Value.zero .fp32 Bool.true ∧
    (widen (FP.Value.ofBits .fp32 127)).isNaN=Bool.true := by
  decide +kernel

/-- Halfway, subnormal, saturation and special-value regression cases. -/
theorem rounding_boundaries :
    narrow (FP.Value.ofBits .fp32 0x3f880000)=FP.Value.ofBits .fp32 56 ∧
    narrow (FP.Value.ofBits .fp32 0x3a800000)=FP.Value.ofBits .fp32 0 ∧
    narrow (FP.Value.ofBits .fp32 0x3b400000)=FP.Value.ofBits .fp32 2 ∧
    narrow (FP.Value.ofBits .fp32 0x43d80000)=FP.Value.ofBits .fp32 126 ∧
    narrow (FP.Value.infinity .fp32)=FP.Value.ofBits .fp32 126 ∧
    narrow (FP.Value.infinity .fp32 Bool.true)=FP.Value.ofBits .fp32 254 ∧
    narrow (FP.Value.zero .fp32 Bool.true)=FP.Value.ofBits .fp32 128 ∧
    narrow (FP.Value.qNaN .fp32)=FP.Value.ofBits .fp32 127 := by
  decide +kernel

/-- Exhaustive finite-encoding round trip, including both signed zeros and
all positive/negative normal and subnormal E4M3 finite values. -/
theorem finite_roundtrip : ∀ (i : Fin 127) (negative : Bool),
    let b := (if negative then 128 else 0)+i.val
    narrow (widen (FP.Value.ofBits .fp32 b))=FP.Value.ofBits .fp32 b := by
  decide +kernel

#axiomsClean decode_boundaries
#axiomsClean rounding_boundaries
#axiomsClean finite_roundtrip
end VeriTile.Bench.TritonBenchG.TritonMatmul.E4M3
