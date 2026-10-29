import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelSource
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Real

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelRealModel
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

/-- A genuine nonempty mathematical maximum, with no artificial zero seed.
The source rejects an empty feature axis before this callback is evaluated. -/
def maximum {n : Nat} (values : Fin n → ℝ) : ℝ :=
  if hn : 0 < n then
    Finset.univ.sup' ⟨⟨0, hn⟩, Finset.mem_univ _⟩ values
  else 0

theorem maximum_bound {n : Nat} (values : Fin n → ℝ) (i : Fin n) :
    values i ≤ maximum values := by
  have hn : 0 < n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
  simpa [maximum, hn] using (Finset.le_sup' (f := values) (Finset.mem_univ i))

theorem maximum_le {n : Nat} (values : Fin n → ℝ) (hn : 0 < n)
    (bound : ℝ) (h : ∀ i, values i ≤ bound) : maximum values ≤ bound := by
  simp only [maximum, hn, dite_true]
  exact Finset.sup'_le _ _ (fun i _ => h i)

theorem maximum_abs {n : Nat} (hn : 0 < n) (values : Nat → ℝ) :
    maximum (fun i : Fin n => |values i.val|) = mathematicalMaximum values n := by
  apply le_antisymm
  · apply maximum_le _ hn
    intro i
    exact RowwiseQuantizationTriton.logicalMaximum_bound values n i.val i.isLt
  · apply RowwiseQuantizationTriton.logicalMaximum_least
    · exact le_trans (abs_nonneg (values 0))
        (maximum_bound (fun i : Fin n => |values i.val|) ⟨0, hn⟩)
    · intro i hi
      exact maximum_bound (fun j : Fin n => |values j.val|) ⟨i, hi⟩

def reduceMax {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → Values ℝ .real shape →
      Values ℝ .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false, values => fun out =>
      maximum (fun i => values (TileShape.insertAxisIndex shape ax out i))
  | Bool.true, values => fun out =>
      maximum (fun i => values (TileShape.replaceAxisIndex shape ax out i))

/-- Mathematical interpretation of the same source. FP32 arithmetic is Real
arithmetic; FP16 rounding is erased; signed-byte conversion remains explicit.
Unused operations retain the caller's fallback and cannot affect this kernel. -/
def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    binary := fun p d op x y => match op with
      | .div => x / y
      | _ => rest.binary p d op x y
    unary := fun p op x => match op with
      | .abs _ => |x|
      | _ => rest.unary p op x
    cast := fun _ _ _ x => x
    fp32Load := id
    reduceMax := fun _ => reduceMax }

def equal (x y : ℝ) : Bool := decide (x = y)
def convert (x : ℝ) : Int := (BitVec.ofInt 8 (truncateReal x)).toInt

/-- Independent signed-byte target, with logical coordinates only. -/
def scale (values : Nat → ℝ) (width : Nat) : ℝ := mathematicalMaximum values width / 127
def code (values : Nat → ℝ) (width column : Nat) : Int :=
  if mathematicalMaximum values width = 0 then 0
  else (BitVec.ofInt 8 (truncateReal (values column / scale values width))).toInt

#axiomsClean maximum_bound
#axiomsClean maximum_le
#axiomsClean maximum_abs
#specNonCircular code avoiding [KernelSource.kernel, exec, sourceRegisters]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelRealModel
