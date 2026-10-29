import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelCoverage
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelRealOutput
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelRealBinding

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelRealComplete
open VeriTile.Triton FP.Structural
open Contract (Inputs InputContract)
open KernelBinding (Bytes Cast Preparation parameters fp8)
open KernelNative (Word)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

abbrev Coordinate (i : Inputs) := KernelGrid.Coordinate (Checked.shape i).B (parameters i)

/-- Independent finite-sum mathematical target, reading the original physical
input words and the original scalar RMS dtype conversion. -/
def target (d32 d16 : Word → ℝ) (d8 : Float8E5Bits → ℝ)
    (i : Inputs) (a : Checked.Allocation i) (cast : Cast) (before : Bytes) (x : Coordinate i) : ℝ :=
  KernelMath.output (parameters i).K (parameters i).headDim (parameters i).start (parameters i).rotary
    (KernelReal.logical d8 (fp8 i) (KernelRealBinding.originalInputs d32 d16 i a cast before)
      x.1.val x.2.1.val) x.2.1.val x.2.2.val

theorem even_columns (i : Inputs) (h : InputContract i) : (parameters i).N%2=0 := by
  have valid := (Checked.facts i h).valid
  change (Checked.shape i).N%2=0
  rw [valid.2.2.2.2,Nat.mul_mod,valid.2.2.2.1]
  simp

/-- Actual source kernel, on the complete original grid, computes the
mathematical RMS projection and optional rotation under exact-Real scalar
semantics. Every cell outside its logical output footprint is preserved. -/
theorem complete (d32 d16 : Word → ℝ) (d8 : Float8E5Bits → ℝ)
    (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (i : Inputs) (h : InputContract i) (a : Checked.Allocation i)
    (cast : Cast) (before prepared : Bytes) (prep : KernelBinding.Preparation i a cast before prepared)
    (scratch : State ℝ) :
    ∃ final, KernelGrid.launch (KernelReal.operations d8) (fp8 i) (Checked.shape i).B (parameters i)
      (KernelRealBinding.state d32 d16 i a prepared scratch)=some final ∧
      (∀ x : Coordinate i, final.mem "OUT" (KernelGrid.flat (parameters i) x)=
        Cell.mk (KernelSource.outputDType (fp8 i)).toTileDType
          (ofFloat (KernelSource.outputDType (fp8 i)) (target d32 d16 d8 i a cast before x))) ∧
      (∀ (r : RegionName) (off : Nat), (r≠"OUT" ∨ ∀ x : Coordinate i,off≠KernelGrid.flat (parameters i) x) →
        final.mem r off=(KernelRealBinding.state d32 d16 i a prepared scratch).mem r off) := by
  obtain ⟨final,run,out,frame⟩ := KernelGrid.logical_complete (KernelReal.operations d8) (fp8 i)
    (Checked.shape i).B (parameters i) (KernelRealBinding.state d32 d16 i a prepared scratch)
    (KernelRealBinding.state_bytes d32 d16 i a prepared scratch)
  refine ⟨final,run,?_,frame⟩
  intro x
  have typed := out x
  unfold KernelGrid.result at typed
  rw [KernelRealBinding.state_inputs] at typed
  have agree := KernelRealBinding.values_agreement d32 d16
    (KernelBinding.original_agreement i h a cast before prepared prep x.1.val x.1.isLt)
  have hm : 0<(parameters i).M := by have := x.2.1.isLt; omega
  have hn : 0<(parameters i).N := by have := x.2.2.isLt; omega
  have graph := KernelGraph.output_congr (KernelReal.operations d8) (fp8 i) (parameters i)
    _ (KernelRealBinding.originalInputs d32 d16 i a cast before)
    (KernelGrid.tile (parameters i) (KernelGrid.owner (parameters i) x)) hm hn agree (KernelGrid.lane x)
  rw [graph] at typed
  obtain ⟨hrow,hcol⟩ := KernelGrid.owner_coordinate (parameters i) x
  change (KernelGrid.tile (parameters i) (KernelGrid.owner (parameters i) x)).row*16+
    (KernelGrid.lane x).1.val=x.2.1.val at hrow
  change (KernelGrid.tile (parameters i) (KernelGrid.owner (parameters i) x)).column*64+
    (KernelGrid.lane x).2.1.val=x.2.2.val at hcol
  have real := KernelReal.output_real d8 zero (fp8 i) (parameters i)
    (KernelRealBinding.originalInputs d32 d16 i a cast before)
    (KernelGrid.tile (parameters i) (KernelGrid.owner (parameters i) x)) (KernelGrid.lane x)
    (by rw [hrow]; exact x.2.1.isLt)
    (by rw [hcol]; exact x.2.2.isLt) (even_columns i h)
  rw [real,hrow,hcol] at typed
  exact typed

#axiomsClean even_columns
#axiomsClean complete
#specNonCircular target avoiding [FP.Structural.exec,KernelSource.inputs,KernelSource.result,
  KernelSource.scatter,KernelGraph.output,KernelGrid.result,KernelGrid.launch]
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelRealComplete
