import Mathlib.Data.Nat.Basic
import Init.Data.Nat.Power2
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.MatrixTranspose
open VeriTile.Triton

/-- Row-major mathematical transpose, independent of kernel execution.
Its output coordinate is `(col,row)`, at flat address `col*m+row`. -/
def formula {α : Type} (matrix : Nat → α) (d row col : Nat) : α :=
  matrix (row*d+col)

/-- Equivalent flat output formula: output[o] = matrix[(o % m)*d + o/m]. -/
def flatFormula {α : Type} (matrix : Nat → α) (m d o : Nat) : α :=
  matrix ((o % m)*d+o/m)

/-- The wrapper supplies row-major FP16 storage. This binds each logical
input element to an actual typed memory cell, including every FP16 bit word.
The wrapper has no caller-provided Tensor to inspect; ordinary successful
`torch.randn(..., dtype=torch.float16)` provides this storage interpretation. -/
def InputCells {α : Type} (m d : Nat)
    (memory : RegionName → Nat → FP.Structural.Cell α) (matrix : Nat → α) : Prop :=
  ∀ i, i < m*d → memory "matrix" i = .mk .fp16 (matrix i)

/-- Complete dimension family, independently stated using powers of two. -/
def SourceShape (m d : Nat) : Prop :=
  ∃ a b : Nat, m = 2^a ∧ d = 2^b ∧ a+b ≤ 20

/-- The exact supported shape domain of the original single unmasked tile. -/
structure InputContract (m d : Nat) : Prop where
  positive : 0 < m ∧ 0 < d
  powers : m &&& (m-1) = 0 ∧ d &&& (d-1) = 0
  extent : m*d ≤ 2^20

inductive Argument where
  | integer (value : Int)
  | other
  deriving DecidableEq, Repr

inductive InputError where
  | integers | positive | powers | extent
  deriving DecidableEq, Repr

/-- Integer checks in source order. toNat is used only after positivity. -/
def checkIntegers (m d : Int) : Except InputError Unit :=
  if m ≤ 0 ∨ d ≤ 0 then .error .positive
  else if m.toNat &&& (m.toNat-1) ≠ 0 ∨ d.toNat &&& (d.toNat-1) ≠ 0 then
    .error .powers
  else if 2^20 < m.toNat*d.toNat then .error .extent
  else .ok ()

/-- `integer` means exact built-in Python int, excluding bool and subclasses. -/
def checkInputs (m d : Argument) : Except InputError Unit :=
  match m, d with
  | .integer a, .integer b => checkIntegers a b
  | _, _ => .error .integers

def AdmittedArguments (m d : Argument) : Prop :=
  match m, d with
  | .integer a, .integer b => 0 < a ∧ 0 < b ∧ InputContract a.toNat b.toNat
  | _, _ => False

/-- Optional historical host metadata. Kernel acceptance does not prove
allocation or returned Tensor identity. -/
structure ResultMetadata where
  region : RegionName
  shape : List Nat
  strides : List Nat
  storage : FloatDType
  cuda : Bool

def returnedTensor (m d : Nat) : ResultMetadata :=
  ⟨"out", [d,m], [m,1], .fp16, Bool.true⟩

end VeriTile.Bench.Repairs.MatrixTranspose
