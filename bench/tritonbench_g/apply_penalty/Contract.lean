import VeriTile.Torch.Tensor
import Mathlib.Data.Nat.Log

/-! The checked contiguous FP32/int32 domain. Integer words remain signed
int32; their nonnegative natural projections are used only after admission.
Tensor metadata and integer contents are entry observations, not outputs. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty
open VeriTile.Torch

structure Arguments where
  tensor : Fin 7 → TensorMetadata
  batches : Nat
  vocab : Nat
  hint : Int
  ids : Nat → BitVec 32
  counts : Nat → BitVec 32
  cumsum : Nat → BitVec 32

def sourceBlock (hint : Int) : Nat := 2 ^ max 9 ((hint.toNat - 1).log2 + 1)
def Arguments.start (a : Arguments) (b : Nat) : Nat := (a.cumsum b).toInt.toNat
def Arguments.stop (a : Arguments) (b : Nat) : Nat := (a.cumsum (b+1)).toInt.toNat
def Arguments.token (a : Arguments) (k : Nat) : Nat := (a.ids k).toInt.toNat
def Arguments.count (a : Arguments) (k : Nat) : Nat := (a.counts k).toInt.toNat
def Arguments.block (a : Arguments) : Nat := sourceBlock a.hint
def Arguments.capacity (a : Arguments) (r : Fin 7) : Nat := (a.tensor r).numel

def TensorValid (a : Arguments) (r : Fin 7) : Prop :=
  let t := a.tensor r
  t.layout = .strided ∧
  t.dtype = (if r.val < 4 then .floating .fp32 else .other "int32") ∧
  t.shape.length = (if r.val = 0 then 2 else 1) ∧
  t.device.kind = "cuda" ∧ t.device = (a.tensor 0).device ∧
  t.isContiguous = true ∧ t.isNeg = false ∧ t.numel < 2^31 ∧
  (0 < t.numel → t.dataPtr.toNat % 4 = 0)

def DisjointLogits (a : Arguments) (r : Fin 7) : Prop :=
  a.capacity 0 = 0 ∨ a.capacity r = 0 ∨
  (a.tensor 0).dataPtr.toNat + 4 * a.capacity 0 ≤ (a.tensor r).dataPtr.toNat ∨
  (a.tensor r).dataPtr.toNat + 4 * a.capacity r ≤ (a.tensor 0).dataPtr.toNat

def WindowValid (a : Arguments) (b : Nat) : Prop :=
  0 ≤ (a.cumsum b).toInt ∧
  (a.cumsum b).toInt ≤ (a.cumsum (b+1)).toInt ∧
  a.stop b ≤ min (a.capacity 4) (a.capacity 5) ∧
  a.stop b - a.start b ≤ a.block ∧
  (∀ k, a.start b ≤ k → k < a.stop b →
    0 ≤ (a.ids k).toInt ∧ a.token k < a.vocab ∧ 0 ≤ (a.counts k).toInt) ∧
  (∀ j k, a.start b ≤ j → j < a.stop b →
    a.start b ≤ k → k < a.stop b → a.token j = a.token k → j = k)

structure InputContract (a : Arguments) : Prop where
  tensors : ∀ r, TensorValid a r
  shape : (a.tensor 0).shape = [a.batches, a.vocab]
  hint : 0 ≤ a.hint ∧ a.hint ≤ 65536
  penalties : ∀ r : Fin 7, 1 ≤ r.val → r.val < 4 → a.batches ≤ a.capacity r
  cumsum : 0 < a.batches → a.batches + 1 ≤ a.capacity 6
  disjoint : ∀ r : Fin 7, 0 < r.val → DisjointLogits a r
  windows : ∀ b, b < a.batches → WindowValid a b

inductive InputError where
  | tensors | shape | hint | penalties | cumsum | alias | window
  deriving DecidableEq, Repr

/-- Source-ordered checks over observed Tensor attributes and host integer
reads. Attribute classification precedes this typed admission function. -/
noncomputable def checkInputs (a : Arguments) : Except InputError Unit := by
  classical
  exact if ¬ (∀ r, TensorValid a r) then .error .tensors
    else if (a.tensor 0).shape ≠ [a.batches, a.vocab] then .error .shape
    else if ¬ (0 ≤ a.hint ∧ a.hint ≤ 65536) then .error .hint
    else if ¬ (∀ r : Fin 7, 1 ≤ r.val → r.val < 4 → a.batches ≤ a.capacity r)
      then .error .penalties
    else if ¬ (0 < a.batches → a.batches + 1 ≤ a.capacity 6) then .error .cumsum
    else if ¬ (∀ r : Fin 7, 0 < r.val → DisjointLogits a r) then .error .alias
    else if ¬ (∀ b, b < a.batches → WindowValid a b) then .error .window
    else .ok ()

end VeriTile.Bench.Repairs.ApplyPenalty
