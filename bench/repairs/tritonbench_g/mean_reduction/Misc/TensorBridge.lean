import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelReal
import bench.tritonbench_g.mean_reduction.Proofs.SourceFormula

/-! Original-coordinate and compressed-Tensor applications of the complete raw
kernel theorems. Existing compression and storage premises are retained. -/
namespace VeriTile.Bench.Repairs.MeanReduction.TensorBridge
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open VeriTile.Bench.TritonBenchG.MeanReduction Whole
noncomputable section
local instance : Inhabited VeriTile.Torch.FP32.Word := ⟨FP.Value.zero .fp32⟩

/-- Every launched kernel writes the independent finite mean of the original
 logical coordinates. The source's compression is an input-storage contract;
 wrapper execution, allocation and output shape handling are not proof goals.
 Arbitrary complete program order, empty batches and the full frame are covered. -/
theorem mean_dim_kernel_correct (x : Tensor rank) (dims : List (Fin rank))
    (h : InputContract metadata x.shape dims dtype) (_binding : TensorBinding metadata x)
    (entry : Nat → ℝ) (s : BlockState)
    (input : KernelReal.InputValues x dims h.distinct entry s)
    (tiles : List Nat) (schedule : tiles.Perm (List.range (Whole.gridSize (rows x.shape dims)))) :
    VeriTile.Spec.Real (∃ final,
      Whole.runTiles "X" "Mean" (rows x.shape dims) (columns x.shape dims) tiles s = some final ∧
      (∀ row : Fin (rows x.shape dims), final.readMem "Mean" row.val =
        mathematicalMean x.shape dims (x.read entry) ((batchLinear x.shape dims).symm row)) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ rows x.shape dims ≤ o → final.mem r o = s.mem r o)) :=
  KernelReal.coordinates x dims h.distinct entry s input tiles schedule

/-- The complete source grid reads the same physical FP32 Tensor checked by
 the core storage laws. This Tensor is the existing compression result, not a
 newly introduced adaptation. All exterior cells are preserved. -/
theorem mean_kernel_fp32 (family : Nat → Algebra VeriTile.Torch.FP32.Word)
    (x : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws x)
    (contiguous : x.metadata.isContiguous = Bool.true) (noNeg : x.metadata.isNeg = Bool.false)
    (M N : Nat) (shape : x.metadata.numel = M*N)
    (read : Nat → VeriTile.Torch.FP32.Word) (s : State VeriTile.Torch.FP32.Word)
    (input : KernelFP.InputCells x laws contiguous read s)
    (tiles : List Nat) (schedule : tiles.Perm (List.range (Whole.gridSize M))) :
    ∃ t, WholeFPFamily.runTiles family Bool.false "X" "Mean" M N tiles s = some t ∧
      WholeFPFamily.runTiles family Bool.true "X" "Mean" M N tiles s = some t ∧
      (∀ i, i < M → t.mem "Mean" i = .mk .real
        (WholeFPFamily.rowValue family "X" M N
          (fun _ j => .mk .real (VeriTile.Torch.FP32.logicalValue read x laws j)) i)) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hc, hv, hf⟩ := KernelNativeResult.mean_grid_fp32 family M N s tiles schedule
  refine ⟨t,he,hc,?_,hf⟩
  intro i hi
  rw [hv i hi]
  apply congrArg (Cell.mk .real)
  apply WholeFPValue.rowValue_congr_active
  intro j hj
  rw [input j (by omega), Cell.read_same,
    VeriTile.Torch.FP32.direct_read_eq_logical read x laws contiguous noNeg j (by omega)]
  rfl

/-- The existing compression API copies original coordinates in its actual
 stride-sorted order. This binds input words only; it assumes no mean result. -/
def CompressionValues (original : Whole.Tensor rank) (dims : List (Fin rank))
    (h : InputContract metadata original.shape dims dtype)
    (entry : Nat → VeriTile.Torch.FP32.Word)
    (buffer : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws buffer)
    (read : Nat → VeriTile.Torch.FP32.Word) : Prop :=
  ∀ i, i < Whole.rows original.shape dims * Whole.columns original.shape dims →
    VeriTile.Torch.FP32.logicalValue read buffer laws i =
      Whole.flattenedInput original dims h.distinct entry h.positive i

/-- Native primitive graph on the same original Tensor coordinates used by
 Correct. Only the already-existing compression is supplied as an API binding;
 all source kernel execution, output values and exterior frame are proved. -/
theorem mean_original_fp32 (family : Nat → Algebra VeriTile.Torch.FP32.Word)
    (original : Whole.Tensor rank) (dims : List (Fin rank))
    (h : InputContract metadata original.shape dims dtype) (_binding : TensorBinding metadata original)
    (entry : Nat → VeriTile.Torch.FP32.Word)
    (buffer : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws buffer)
    (contiguous : buffer.metadata.isContiguous = Bool.true) (noNeg : buffer.metadata.isNeg = Bool.false)
    (shape : buffer.metadata.numel = Whole.rows original.shape dims * Whole.columns original.shape dims)
    (read : Nat → VeriTile.Torch.FP32.Word) (s : State VeriTile.Torch.FP32.Word)
    (input : KernelFP.InputCells buffer laws contiguous read s)
    (copy : CompressionValues original dims h entry buffer laws read)
    (tiles : List Nat) (schedule : tiles.Perm (List.range (Whole.gridSize (Whole.rows original.shape dims)))) :
    ∃ t, WholeFPFamily.runTiles family Bool.false "X" "Mean"
        (Whole.rows original.shape dims) (Whole.columns original.shape dims) tiles s = some t ∧
      WholeFPFamily.runTiles family Bool.true "X" "Mean"
        (Whole.rows original.shape dims) (Whole.columns original.shape dims) tiles s = some t ∧
      (∀ i, i < Whole.rows original.shape dims → t.mem "Mean" i = .mk .real
        (Whole.fpFormula (family (i/8)) original dims h.distinct entry h.positive i)) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ Whole.rows original.shape dims ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t,he,hc,hv,hf⟩ := mean_kernel_fp32 family buffer laws contiguous noNeg
    (Whole.rows original.shape dims) (Whole.columns original.shape dims) shape read s input tiles schedule
  refine ⟨t,he,hc,?_,hf⟩
  intro i hi
  rw [hv i hi]
  apply congrArg (Cell.mk .real)
  apply WholeFPValue.rowValue_congr_active
  intro j hj
  exact copy j hj

#axiomsClean mean_dim_kernel_correct
#axiomsClean mean_kernel_fp32
#axiomsClean mean_original_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.TensorBridge
