import bench.tritonbench_g.kv_cache_filling.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantProfile
open VeriTile.Triton FP.Structural
noncomputable section

/-- Scalar primitives and complete ordered lane reductions for the FP32
source. Reductions receive every lane, including the explicit padding. -/
structure Ops (α : Type) where
  literal : ℝ → α
  add : α → α → α
  sub : α → α → α
  divide : α → α → α
  minimum : α → α → α
  maximum : α → α → α
  equal : α → α → Bool
  byte : α → BitVec 8
  low : (n : Nat) → (Fin n → α) → α
  high : (n : Nat) → (Fin n → α) → α

def reduceAxis {α : Type} (op : (n : Nat) → (Fin n → α) → α)
    {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → Values α .real shape →
      Values α .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false, xs => fun i => op _ (fun k => xs (TileShape.insertAxisIndex shape ax i k))
  | Bool.true, xs => fun i => op _ (fun k => xs (TileShape.replaceAxisIndex shape ax i k))

/-- Unused total operations have explicit inert values. Source-used floating
operations remain independent callbacks; integer conversion is the selected
total byte profile. No output, execution or memory callback is accepted. -/
def algebra {α : Type} (a : Ops α) : Algebra α where
  literal := fun _ _ x => a.literal x
  negInf := a.literal 0
  binary := fun _ _ op x y => match op with
    | .add => a.add x y
    | .sub => a.sub x y
    | .div => a.divide x y
    | .max => a.maximum x y
    | .mul | .pow => a.literal 0
  unary := fun _ op x => match op with
    | .neg _ => a.sub (a.literal 0) x
    | _ => a.literal 0
  compareEq := fun _ _ => some a.equal
  minimum := fun _ _ => some a.minimum
  cast := fun _ _ _ x => x
  fromNat := fun _ n => a.literal n
  fromInt := fun _ n => a.literal n
  fp32Bits := fun _ => a.literal 0
  fp32Load := id
  floatToNat := fun _ _ width =>
    if width = 8 then some (fun x => some (a.byte x).toNat) else none
  reduceMin := fun _ {_} ax keepDims xs => some (reduceAxis a.low ax keepDims xs)
  reduceMax := fun _ => reduceAxis a.high
  reduceSum := fun _ {_} _ _ _ _ => a.literal 0

def pad {α : Type} (w n : Nat) (x : Nat → α) : Fin n → α :=
  fun i => x (if i.val < w then i.val else 0)

structure Row (α : Type) where
  values : Nat → BitVec 8
  scale : α
  zero : α

def scale {α : Type} (a : Ops α) (levels : Nat) (lo hi : α) : α :=
  if a.equal hi lo then a.literal 1 else a.divide (a.sub hi lo) (a.literal levels)

def zero {α : Type} (a : Ops α) (levels : Nat) (lo hi : α) : α :=
  a.divide (a.sub (a.literal 0) lo) (scale a levels lo hi)

def code {α : Type} (a : Ops α) (levels : Nat) (lo hi x : α) : BitVec 8 :=
  a.byte (a.add (a.add (a.divide x (scale a levels lo hi)) (zero a levels lo hi))
    (a.literal (1 / 2)))

def row8 {α : Type} (a : Ops α) (w n : Nat) (x : Nat → α) : Row α :=
  let lo := a.low n (pad w n x)
  let hi := a.high n (pad w n x)
  ⟨fun i => code a 255 lo hi (x i), scale a 255 lo hi, zero a 255 lo hi⟩

def row4 {α : Type} (a : Ops α) (w n : Nat) (x : Nat → α) : Row α :=
  let left := pad w n x
  let right := pad w n (fun i => x (w + i))
  let lo := a.low n (fun i => a.minimum (left i) (right i))
  let hi := a.high n (fun i => a.maximum (left i) (right i))
  ⟨fun i => code a 15 lo hi (x i) + code a 15 lo hi (x (w + i)) * 16,
    scale a 15 lo hi, zero a 15 lo hi⟩

end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantProfile
