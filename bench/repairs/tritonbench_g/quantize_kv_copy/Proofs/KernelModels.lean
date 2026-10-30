import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelExecution
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Real
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelModels
open VeriTile.Triton FP.Structural Kernel
noncomputable section
open Classical

/-- The selected local reduction tree has exactly the eight source lanes.
Its compiler realization is an explicit local layout/instruction boundary. -/
structure ReductionTree where
  tree : Tree
  covers : tree.leaves.Perm (List.range 8)

/-- Fixed-arity native instruction semantics. All FP32 words are admitted,
including nonfinite quotients and underflowed scales. Conversion is the explicit
native scalar-to-byte instruction profile, never a finite-input restriction. -/
structure NativeInstructions where
  absolute : FP.Value .fp32 → FP.Value .fp32
  maximum : FP.Value .fp32 → FP.Value .fp32 → FP.Value .fp32
  divide : FP.Value .fp32 → FP.Value .fp32 → FP.Value .fp32
  equal : FP.Value .fp32 → FP.Value .fp32 → Bool
  int8 : FP.Value .fp32 → BitVec 8

abbrev Word := FP.Value .fp32
def zero : Word := FP.Value.ofBits .fp32 0
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def levels : Word := FP.Value.ofBits .fp32 0x42fe0000

def nativeMaximum (a : NativeInstructions) (tree : ReductionTree) {n : Nat} (x : Fin n → Word) : Word :=
  tree.tree.evaluate a.maximum (fun i => if h : i<n then x ⟨i,h⟩ else zero)

def nativeOps (a : NativeInstructions) (tree : ReductionTree) : Ops Word where
  zero := zero
  one := one
  levels := levels
  abs := a.absolute
  divide := a.divide
  equal := a.equal
  maximum := nativeMaximum a tree
  convert := fun x => (a.int8 x).toInt

/-- Independent indexed target, without source syntax, execution, registers,
stores, final memories or a complete result premise. FP32 scales remain FP32. -/
def nativeMax (a : NativeInstructions) (tree : ReductionTree) (x : Nat → Word) : Word :=
  tree.tree.evaluate a.maximum (fun i => a.absolute (x i))
def nativeScale (a : NativeInstructions) (tree : ReductionTree) (x : Nat → Word) : Word :=
  a.divide (nativeMax a tree x) levels
def nativeCode (a : NativeInstructions) (tree : ReductionTree) (x : Nat → Word) (lane : Nat) : BitVec 8 :=
  a.int8 (a.divide (x lane)
    (if a.equal (nativeMax a tree x) zero then one else nativeScale a tree x))

theorem native_maximum (a : NativeInstructions) (tree : ReductionTree) (x : Nat → Word) :
    (nativeOps a tree).maximum (fun i : Fin 8 => (nativeOps a tree).abs (x i.val))=nativeMax a tree x := by
  apply RowwiseQuantizationTriton.Tree.evaluate_congr
  intro i hi
  have hb : i<8 := List.mem_range.mp (tree.covers.mem_iff.mp hi)
  simp [nativeOps,nativeMaximum,nativeMax,hb]

/-- Full mathematical maximum, with no artificial zero seed on nonempty axes. -/
def realMaximum {n : Nat} (x : Fin n → ℝ) : ℝ :=
  if h : 0<n then Finset.univ.sup' ⟨⟨0,h⟩,Finset.mem_univ _⟩ x else 0

theorem realMaximum_bound {n : Nat} (x : Fin n → ℝ) (i : Fin n) : x i≤realMaximum x := by
  have h : 0<n := Nat.zero_lt_of_lt i.isLt
  simpa [realMaximum,h] using (Finset.le_sup' (f:=x) (Finset.mem_univ i))

theorem realMaximum_le {n : Nat} (x : Fin n → ℝ) (h : 0<n) (bound : ℝ)
    (hb : ∀ i,x i≤bound) : realMaximum x≤bound := by
  simp only [realMaximum,h,dite_true]
  exact Finset.sup'_le _ _ (fun i _ => hb i)

theorem realMaximum_abs (x : Nat → ℝ) :
    realMaximum (fun i : Fin 8 => |x i.val|)=mathematicalMaximum x 8 := by
  apply le_antisymm
  · exact realMaximum_le _ (by decide) _ (fun i => RowwiseQuantizationTriton.logicalMaximum_bound x 8 i.val i.isLt)
  · apply RowwiseQuantizationTriton.logicalMaximum_least
    · exact le_trans (abs_nonneg (x 0)) (realMaximum_bound (fun i : Fin 8 => |x i.val|) ⟨0,by decide⟩)
    · intro i hi
      exact realMaximum_bound (fun j : Fin 8 => |x j.val|) ⟨i,hi⟩

def realOps : Ops ℝ where
  zero := 0
  one := 1
  levels := 127
  abs := abs
  divide := (· / ·)
  equal := fun x y => decide (x=y)
  maximum := realMaximum
  convert := fun x => (BitVec.ofInt 8 (truncateReal x)).toInt

def realScale (x : Nat → ℝ) : ℝ := mathematicalMaximum x 8/127
def realCode (x : Nat → ℝ) (lane : Nat) : BitVec 8 :=
  if mathematicalMaximum x 8=0 then 0
  else BitVec.ofInt 8 (truncateReal (x lane/realScale x))

theorem source_literals : zero.toRat?=some 0 ∧ one.toRat?=some 1 ∧ levels.toRat?=some 127 := by
  decide +kernel

#axiomsClean native_maximum
#axiomsClean realMaximum_abs
#axiomsClean source_literals
#specNonCircular nativeScale avoiding [kernel,FP.Structural.exec]
#specNonCircular nativeCode avoiding [kernel,FP.Structural.exec]
#specNonCircular realScale avoiding [kernel,FP.Structural.exec]
#specNonCircular realCode avoiding [kernel,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelModels
