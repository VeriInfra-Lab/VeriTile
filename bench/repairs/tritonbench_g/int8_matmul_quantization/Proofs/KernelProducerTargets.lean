import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelProducerGrid
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelScalarProfile
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Ideal

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducerTargets
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
abbrev Tree := KernelScalarProfile.Tree
abbrev Geometry := KernelProducerGrid.Geometry

/-- A structural reduction network covering the launched lanes exactly.
Its choice depends on the source launch, never on input values. -/
structure Selection (p : Geometry) where
  tree : Tree
  complete : tree.leaves.Perm (List.range p.width)

def mathematicalScale (values : Nat → ℝ) (columns : Nat) : ℝ := maximumAbs values columns / 127
def mathematicalByte (values : Nat → ℝ) (columns column : Nat) : Byte :=
  realInteger8 (values column / mathematicalScale values columns)

def nativeMaximum (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (values : Nat → Word32) : Word32 :=
  f.maximum (FP.Value.zero .fp32) (selection.tree.evaluate f.maximum
    (fun c => if c < p.columns then (values c).abs else FP.Value.zero .fp32))
def nativeScale (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (values : Nat → Word32) : Word32 :=
  f.divide (nativeMaximum f p selection values) (FP.Value.ofBits .fp32 0x42fe0000)
def nativeByte (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (values : Nat → Word32) (column : Nat) : Byte :=
  KernelNativeModel.integer8 (f.divide (values column) (nativeScale f p selection values))
def nativeStoredScale (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (values : Nat → Word32) : Word16 :=
  Int8DequantMatmul.narrow16 (nativeScale f p selection values)

/-- Mathematical reduction closure, including explicit zero initialization
and masked zero lanes. No numerical maximum/reduction law is assumed. -/
theorem mathematical_maximum (p : Geometry) (selection : Selection p)
    (values : Nat → ℝ) (covers : p.columns ≤ p.width) :
    max 0 (selection.tree.evaluate max (fun c => if c < p.columns then |values c| else 0)) =
      maximumAbs values p.columns := by
  apply le_antisymm
  · apply max_le (maximumAbs_nonnegative values p.columns)
    apply RowwiseQuantizationTriton.Tree.maximum_upper
    intro c _
    by_cases hc : c < p.columns
    · simpa [hc] using maximumAbs_covers values p.columns c hc
    · simpa [hc] using maximumAbs_nonnegative values p.columns
  · apply maximumAbs_le values p.columns _ (le_max_left _ _)
    intro c hc
    have member : c ∈ selection.tree.leaves :=
      selection.complete.mem_iff.mpr (List.mem_range.mpr (lt_of_lt_of_le hc covers))
    have bound := RowwiseQuantizationTriton.Tree.maximum_member selection.tree
      (fun j => if j < p.columns then |values j| else 0) c member
    exact le_trans (by simpa [hc] using bound) (le_max_right _ _)

/-- Input-only interpretation; no source execution or output equation. -/
def Reads {α : Type} [Inhabited α] (p : Geometry) (values : Nat → Nat → α)
    (initial : State α) : Prop :=
  ∀ r c, r < p.rows → c < p.columns →
    (initial.mem "x" (r*p.columns+c)).read .real = values r c

#axiomsClean mathematical_maximum
#specNonCircular mathematicalByte avoiding [KernelSource.quantize, KernelProducerGrid.run]
#specNonCircular nativeByte avoiding [KernelSource.quantize, KernelProducerGrid.run]
#specNonCircular Reads avoiding [KernelSource.quantize, KernelProducerGrid.run]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducerTargets
