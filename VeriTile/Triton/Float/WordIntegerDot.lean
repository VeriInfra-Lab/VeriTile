import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! A concrete software model for signed int8 products accumulated modulo 2^32.
The input range is checked. This does not infer operand widths from an erased
Int channel or assert a compiler/native binding. A source contract must retain
its int8 input casts/storage and int32 accumulator wrap at every update. -/
namespace VeriTile.Triton.FP.IntegerDot
open Structural
noncomputable section

def signed8 (x : Int) : Prop := -128 ≤ x ∧ x < 128

def valid {α : Type} {shape : TileShape} (x : Values α .int shape) : Prop :=
  ∀ i, signed8 (x i)

/-- Only integer output and otherwise default options are implemented. -/
def supported (o : DotOptions) : Prop :=
  (o.outDType = none ∨ o.outDType = some .int) ∧ o.allowTF32 = none ∧
    o.inputPrecision = none ∧ o.maxNumImpreciseAcc = none

/-- Explicit finite machine-word fold; no unbounded-Int accumulation. -/
def wordDot : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → Int) → (TileIndex (batch ++ [k,n]) → Int) →
    TileIndex (batch ++ [m,n]) → BitVec 32
  | [], _, k, _, a, b => fun (i,j,_) => (List.finRange k).foldl
      (fun acc t => acc + BitVec.ofInt 32 (a (i,t,PUnit.unit)) *
        BitVec.ofInt 32 (b (t,j,PUnit.unit))) (BitVec.ofNat 32 0)
  | _ :: batch, _, _, _, a, b => fun (i,rest) =>
      wordDot batch (fun t => a (i,t)) (fun t => b (i,t)) rest

def backend {α : Type} (rest : Algebra α) : Algebra α := by
  classical
  refine { rest with integerDot := ?_ }
  intro o batch m k n a b
  exact if (supported o ∧ valid a ∧ valid b) then
    (some (fun i => (wordDot batch a b i).toInt)) else none

@[simp] theorem backend_supported {α : Type} (rest : Algebra α) (o : DotOptions)
    {batch : TileShape} {m k n : Nat} (a : Values α .int (batch ++ [m,k]))
    (b : Values α .int (batch ++ [k,n]))
    (ho : supported o) (ha : valid a) (hb : valid b) :
    (backend rest).integerDot o a b = some (fun i => (wordDot batch a b i).toInt) := by
  classical
  simp [backend, ho, ha, hb]

@[simp] theorem backend_rejects_invalid {α : Type} (rest : Algebra α) (o : DotOptions)
    {batch : TileShape} {m k n : Nat} (a : Values α .int (batch ++ [m,k]))
    (b : Values α .int (batch ++ [k,n])) (h : ¬ (valid a ∧ valid b)) :
    (backend rest).integerDot o a b = none := by
  classical
  simp [backend, show ¬ (supported o ∧ valid a ∧ valid b) from fun hx => h hx.2]

/-- Independent signed integer sum projected to its low 32 bits. -/
theorem fold_is_low32 {ι : Type} (xs : List ι) (a b : ι → Int) (initial : Int) :
    xs.foldl (fun acc i => acc + BitVec.ofInt 32 (a i) * BitVec.ofInt 32 (b i))
      (BitVec.ofInt 32 initial) =
    BitVec.ofInt 32 (xs.foldl (fun acc i => acc + a i * b i) initial) := by
  induction xs generalizing initial with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons]
    rw [← BitVec.ofInt_mul, ← BitVec.ofInt_add]
    exact ih _

/-- Overflow is visible in the concrete word model; it is not ruled away. -/
theorem accumulation_wraps :
    (BitVec.ofInt 32 2147483647 + BitVec.ofInt 32 1).toInt = -2147483648 := by decide

theorem signed_product_nonzero :
    wordDot [] (m := 1) (k := 1) (n := 1) (fun _ => -128) (fun _ => 127)
      (0,0,PUnit.unit) = BitVec.ofInt 32 (-16256) := by decide

#axiomsClean backend_supported
#axiomsClean backend_rejects_invalid
#axiomsClean fold_is_low32
#axiomsClean accumulation_wraps
#axiomsClean signed_product_nonzero
end
end VeriTile.Triton.FP.IntegerDot
