import VeriTile.Triton.Float.ScalarOps

/-!
Canonical deterministic Philox4x32 and fp32 uniform conversion used by Triton
3.7.1. Integer operations retain their modular widths, both seed key words, and
the selected offset width and round count. Uniform conversion uses the existing
nearest-even FP.round and FP.mul operations, including their two rounding steps.
This core module has no Core.Ast, decoder-compatibility, or proof-module dependency.
The corresponding finite/range proofs live in Random.Proofs; regression vectors
live in bench/tests/PhiloxRandom.lean. Hardware conformance is a separate boundary.
-/
namespace VeriTile.Triton.Random

abbrev Word := BitVec 32

structure Counter where
  c0 : Word
  c1 : Word
  c2 : Word
  c3 : Word
  deriving Repr, DecidableEq

structure Key where
  k0 : Word
  k1 : Word
  deriving Repr, DecidableEq

def mulHi (a b : Word) : Word :=
  BitVec.ofNat 32 (a.toNat * b.toNat / 2^32)

def step (c : Counter) (k : Key) : Counter :=
  { c0 := mulHi 0xcd9e8d57 c.c2 ^^^ c.c1 ^^^ k.k0
    c1 := 0xcd9e8d57 * c.c2
    c2 := mulHi 0xd2511f53 c.c0 ^^^ c.c3 ^^^ k.k1
    c3 := 0xd2511f53 * c.c0 }

def bumpKey (k : Key) : Key :=
  { k0 := k.k0 + 0x9e3779b9, k1 := k.k1 + 0xbb67ae85 }

def philox : Nat → Counter → Key → Counter
  | 0, c, _ => c
  | n+1, c, k => philox n (step c k) (bumpKey k)

inductive OffsetWidth where
  | bits32 | bits64
  deriving Repr, DecidableEq

/-- Int-to-uint64 conversion, including negative signed seeds, is modulo 2^64. -/
def seedKey (seed : Int) : Key :=
  let s := (BitVec.ofInt 64 seed).toNat
  { k0 := BitVec.ofNat 32 s, k1 := BitVec.ofNat 32 (s / 2^32) }

/-- Triton's `randint4x` always calls Philox4x32. With a 64-bit offset, the
second 32-bit counter word holds the offset's upper half. It does not switch
to the separate Philox4x64 generator exposed by direct `philox` calls. -/
def offsetCounter (width : OffsetWidth) (offset : Int) : Counter :=
  let s := (BitVec.ofInt 64 offset).toNat
  { c0 := BitVec.ofNat 32 s
    c1 := match width with
      | .bits32 => 0
      | .bits64 => BitVec.ofNat 32 (s / 2^32)
    c2 := 0, c3 := 0 }

def randint (seed offset : Int) (width : OffsetWidth := .bits32)
    (nRounds : Nat := 10) : Word :=
  (philox nRounds (offsetCounter width offset) (seedKey seed)).c0

/-- Equivalent to signed bitcast, then `where(x < 0, -x - 1, x)`.
The range is 0 through 2^31-1, including both endpoints. -/
def foldSign (x : Word) : Nat :=
  if x.toNat < 2^31 then x.toNat else 2^32 - 1 - x.toNat

/-- The fp32 literal actually emitted for 4.6566127342e-10. -/
def scale : FP.Value .fp32 := FP.Value.ofBits .fp32 0x2fffffff

/-- First convert the folded integer with round-to-nearest-even to fp32,
then multiply by the fp32 scale and round again. Exact rational software
arithmetic is used; this definition does not assert hardware conformance. -/
def uniform (word : Word) : FP.Value .fp32 :=
  FP.mul FP.Config.ieeeValue
    (FP.round .fp32 FP.Config.ieeeValue (Rat.ofInt (foldSign word))) scale

def rand (seed offset : Int) (width : OffsetWidth := .bits32)
    (nRounds : Nat := 10) : FP.Value .fp32 :=
  uniform (randint seed offset width nRounds)

end VeriTile.Triton.Random
