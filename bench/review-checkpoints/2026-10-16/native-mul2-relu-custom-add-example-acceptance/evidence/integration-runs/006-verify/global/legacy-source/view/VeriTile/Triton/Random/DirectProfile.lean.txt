import VeriTile.Triton.Random.Direct
import VeriTile.Triton.Random.SourceProfile

/-! Source profiles and validation for direct Philox4x32. The four counter
operands keep their original shapes until their source round expressions
broadcast them. Every source value is checked before that broadcasting. -/
namespace VeriTile.Triton.Random.Direct

inductive WordType where
  | i32 | u32
  deriving Repr, BEq, DecidableEq

def WordType.inRange : WordType → Int → Bool
  | .i32 => IntegerType.i32.inRange
  | .u32 => IntegerType.u32.inRange

structure Profile where
  seedType : IntegerType
  c0Type : WordType
  c1Type : WordType
  c2Type : WordType
  c3Type : WordType
  deriving Repr, BEq, DecidableEq

structure Inputs (s : Shapes) where
  c0 : TileIndex s.c0 → Int
  c1 : TileIndex s.c1 → Int
  c2 : TileIndex s.c2 → Int
  c3 : TileIndex s.c3 → Int

def Inputs.words {s : Shapes} (v : Inputs s) : Words s :=
  ⟨fun i => BitVec.ofInt 32 (v.c0 i), fun i => BitVec.ofInt 32 (v.c1 i),
   fun i => BitVec.ofInt 32 (v.c2 i), fun i => BitVec.ofInt 32 (v.c3 i)⟩

def Profile.inputsValid {s : Shapes} (cfg : Profile) (seed : Int) (v : Inputs s) : Bool :=
  cfg.seedType.inRange seed &&
    (TileShape.allIndices s.c0).all (fun i => cfg.c0Type.inRange (v.c0 i)) &&
    (TileShape.allIndices s.c1).all (fun i => cfg.c1Type.inRange (v.c1 i)) &&
    (TileShape.allIndices s.c2).all (fun i => cfg.c2Type.inRange (v.c2 i)) &&
    (TileShape.allIndices s.c3).all (fun i => cfg.c3Type.inRange (v.c3 i))

/-- A pure projection validates every input, even when the selected result has
no elements. Source expression evaluation is performed by each interpreter
before calling this function. -/
def evalField {s out : Shapes} {n : Nat} (cfg : Profile)
    (plan : Plan s n out) (field : Field) (seed : Int) (v : Inputs s) :
    Option (TileIndex (out.at field) → Nat) :=
  if cfg.inputsValid seed v then
    some (fun i => ((run plan v.words (seedKey seed)).at field i).toNat)
  else none

def uniformInputsValid {shape : TileShape} (v : TileIndex shape → Nat) : Bool :=
  (TileShape.allIndices shape).all (fun i => decide (v i < 2^32))

end VeriTile.Triton.Random.Direct
