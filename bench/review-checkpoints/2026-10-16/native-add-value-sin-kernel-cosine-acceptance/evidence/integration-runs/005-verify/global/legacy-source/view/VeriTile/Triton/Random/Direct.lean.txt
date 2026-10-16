import VeriTile.Triton.Core.Shape
import VeriTile.Triton.Random.Philox

/-!
Shape-preserving Philox4x32 rounds for a scalar seed key. Each output retains
the shape produced by the source round expressions. In particular, zero rounds
return the four original shapes, without first broadcasting them to one shape.
Words are already 32-bit values here; source operand evaluation/range checks,
signed casts and tuple-assignment capture are separate frontend obligations.
-/
namespace VeriTile.Triton.Random.Direct


inductive Field where
  | c0 | c1 | c2 | c3
  deriving Repr, DecidableEq

structure Shapes where
  c0 : TileShape
  c1 : TileShape
  c2 : TileShape
  c3 : TileShape
  deriving Repr, DecidableEq

def Shapes.at (s : Shapes) : Field → TileShape
  | .c0 => s.c0
  | .c1 => s.c1
  | .c2 => s.c2
  | .c3 => s.c3

@[ext] structure Words (s : Shapes) where
  c0 : TileIndex s.c0 → Word
  c1 : TileIndex s.c1 → Word
  c2 : TileIndex s.c2 → Word
  c3 : TileIndex s.c3 → Word

def Words.at {s : Shapes} (v : Words s) : (f : Field) → TileIndex (s.at f) → Word
  | .c0 => v.c0
  | .c1 => v.c1
  | .c2 => v.c2
  | .c3 => v.c3

/-- The source swaps counter dependencies in each round. Products retain their
operand's shape; only the two XOR chains broadcast their counter operands.
The scalar key does not change any result shape. -/
inductive Plan : Shapes → Nat → Shapes → Type where
  | done {s : Shapes} : Plan s 0 s
  | step {s : Shapes} {a b : TileShape} {n : Nat} {out : Shapes}
      (bc21 : Broadcast s.c2 s.c1 a) (bc03 : Broadcast s.c0 s.c3 b)
      (rest : Plan ⟨a, s.c2, b, s.c0⟩ n out) : Plan s (n+1) out

def round {s : Shapes} {a b : TileShape}
    (bc21 : Broadcast s.c2 s.c1 a) (bc03 : Broadcast s.c0 s.c3 b)
    (v : Words s) (key : Key) : Words ⟨a, s.c2, b, s.c0⟩ :=
  { c0 := fun i => mulHi 0xcd9e8d57 (v.c2 (bc21.leftIndex i)) ^^^
      v.c1 (bc21.rightIndex i) ^^^ key.k0
    c1 := fun i => 0xcd9e8d57 * v.c2 i
    c2 := fun i => mulHi 0xd2511f53 (v.c0 (bc03.leftIndex i)) ^^^
      v.c3 (bc03.rightIndex i) ^^^ key.k1
    c3 := fun i => 0xd2511f53 * v.c0 i }

def run {s out : Shapes} {n : Nat} : Plan s n out → Words s → Key → Words out
  | .done, v, _ => v
  | .step bc21 bc03 rest, v, key => run rest (round bc21 bc03 v key) (bumpKey key)

end VeriTile.Triton.Random.Direct
