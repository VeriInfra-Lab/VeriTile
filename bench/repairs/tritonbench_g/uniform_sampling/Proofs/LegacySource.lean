import VeriTile.Triton.Core.Ast
import VeriTile.Triton.Random.Direct

/-! Source operations for one lane/field of the repaired FP32 kernel. All four
Philox10 words precede the stores; indices are widened before products, low
counter addition is uint32, and masked store pointers clamp to N-1. -/
namespace VeriTile.Bench.Repairs.UniformSampling.Source
open VeriTile.Triton

structure Operations (α : Type) where
  extract : Float64Bits → α
  uniform32 : Random.Word → α
  sub32 : α → α → α
  fma32 : α → α → α → α

def words (seed offset : Int) (index : Nat) : Random.Counter :=
  let low := BitVec.ofInt 32 offset + BitVec.ofNat 32 index
  let high := BitVec.ofInt 32 (offset / (2^32 : Int))
  Random.philox 10 ⟨low,high,0,0⟩ (Random.seedKey seed)

def word (seed offset : Int) (index : Nat) (f : Random.Direct.Field) : Random.Word :=
  let result := words seed offset index
  match f with
  | .c0 => result.c0 | .c1 => result.c1 | .c2 => result.c2 | .c3 => result.c3

structure Registers (α : Type) where
  localIndex : Nat
  pointerIndex : Nat
  active : Bool
  word : Random.Word
  uniform : α
  width : α
  value : α

def lane (M : Operations α) (lower upper : Float64Bits) (seed offset : Int)
    (N B pid fieldIndex laneIndex : Nat) (field : Random.Direct.Field) : Registers α :=
  let index := pid*B+laneIndex
  let w := word seed offset index field
  let u := M.uniform32 w
  let width := M.sub32 (M.extract upper) (M.extract lower)
  let value := M.fma32 u width (M.extract lower)
  let out := (pid*4+fieldIndex)*B+laneIndex
  ⟨out,min out (N-1),decide (out<N),w,u,width,value⟩
end VeriTile.Bench.Repairs.UniformSampling.Source
