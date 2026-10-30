import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPProgram

import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPValues
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- A total local reduction backend. It retains dtype, precision, full input
layout, ax and keepDims. It is neither a GEMV oracle nor an algebraic law. -/
structure Reduction (α : Type) where
  apply : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (ax : Fin shape.length) → (keepDims : Bool) → Values α dtype.toTileDType shape →
    Values α dtype.toTileDType (TileShape.reduceShape shape ax keepDims)

def machine (M : Algebra α) (reduction : Reduction α) : Algebra α :=
  ({M with reduceSumTyped := fun p d {_} ax keep v => some (reduction.apply p d ax keep v)}).withDefaultPrecision .fp32

structure Data (α : Type) where
  input : Nat → α
  weights : Nat → α
  output : Nat → α
  indices : Nat → Int
  other : RegionName → Nat → Cell α
  padding : (dtype : TileDType) → RegionName → Nat → Value α dtype

def Data.cells (data : Data α) (r : RegionName) (off : Nat) : Cell α :=
  if r="X" then .mk .real (data.input off)
  else if r="W" then .mk .fp16 (data.weights off)
  else if r="Y" then .mk .real (data.output off)
  else if r="Indices" then .mk .int (data.indices off)
  else data.other r off

def entry (h : Call) (batch row : Nat) (data : Data α) : State α where
  mem := data.cells
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then row else if ax=1 then batch else 0
  numPids := fun ax => if ax=0 then h.N else if ax=1 then h.batches else 1
  undef := data.padding

def loaded (M : Algebra α) (h : Call) (batch : Nat) (data : Data α) (k : Fin h.BK) : α :=
  M.cast (some .fp32) .real .fp16 (M.fp32Load
    (if h.even=Bool.true ∨ k.val<h.K then data.input (batch*h.K+k.val)
     else M.literal (some .fp32) .real 0))

def weight (M : Algebra α) (h : Call) (slot row : Nat) (data : Data α)
    (n : Fin 256) (k : Fin h.BK) : α :=
  if n.val<1 ∧ k.val<h.K then
    data.weights (slot*(h.N*h.K)+row*h.K+n.val*h.K+k.val)
  else M.cast (some .fp32) .real .fp16 (M.literal (some .fp32) .real 0.0)

def products (M : Algebra α) (h : Call) (batch slot row : Nat) (data : Data α) : Values α .fp16 [256,h.BK] :=
  fun (n,k,_) => M.binary (some .fp32) .fp16 .mul
    (loaded M h batch data k) (weight M h slot row data n k)

def sums (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch slot row : Nat) (data : Data α) : Values α .fp16 [256] :=
  reduction.apply (some .fp32) .fp16 ⟨1,by simp⟩ Bool.false (products M h batch slot row data)

def old (M : Algebra α) (h : Call) (batch row : Nat) (data : Data α) (n : Fin 256) : α :=
  M.fp32Load (if n.val<1 then data.output (batch*h.columns+row+h.slice+n.val)
    else data.padding .real "Y" (batch*h.columns+row+h.slice+n.val))

def result (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch slot row : Nat) (data : Data α) (n : Fin 256) : α :=
  let total := M.cast (some .fp32) .fp16 .real (sums M reduction h batch slot row data (n,PUnit.unit))
  if h.add then M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add total (old M h batch row data n)) else total

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPValues
