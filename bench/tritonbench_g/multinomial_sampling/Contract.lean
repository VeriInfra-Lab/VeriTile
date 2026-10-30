import VeriTile.Torch.Tensor
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace VeriTile.Bench.Repairs.MultinomialSampling.Contract
open scoped BigOperators

/-- Direct Boolean observations queried by the Python `_tensor` guard. The
dtype, shape, device and pointer observations live in the same-call descriptors. -/
structure Flags where
  tensor : Bool
  strided : Bool
  cuda : Bool
  contiguous : Bool
  negative : Bool
  conjugate : Bool

def ordinary (f : Flags) : Bool :=
  f.tensor && f.strided && f.cuda && f.contiguous && !f.negative && !f.conjugate

def shapeCheck (batches tokens : Nat) : Bool :=
  decide (batches < 2^31 ∧ 0 < tokens ∧ tokens < 2^31)

def paddedRows (batches : Nat) := (batches+7)/8*8
def paddedTokens (tokens : Nat) := (tokens+127)/128*128
def last (rows tokens sb st : Nat) := (rows-1)*sb+(tokens-1)*st

def matrixWindow (rows tokens sb st base width : Nat) : Bool :=
  decide (sb < 2^31 ∧ st < 2^31 ∧ last rows tokens sb st < 2^31 ∧
    base % width = 0 ∧ base+width*last rows tokens sb st < 2^64)

def vectorWindow (rows base : Nat) : Bool :=
  decide (base % 8 = 0 ∧ base+8*(rows-1) < 2^64)

/-- A successful finite FP32-to-Python-float read is exact. Its integer ratio
is converted to units of 2^-149, with unbounded Python integer summation. -/
def probabilityRow (tokens : Nat) (finite : Fin tokens → Bool)
    (units : Fin tokens → Int) : Bool :=
  decide ((∀ j, finite j = Bool.true ∧ 0 ≤ units j) ∧ ∑ j, units j = (2:Int)^149)

end VeriTile.Bench.Repairs.MultinomialSampling.Contract
