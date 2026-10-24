import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open Binding
noncomputable section

theorem sum_range (n : Nat) (f : Nat → ℝ) :
    ((List.range n).map f).sum = ∑ j ∈ Finset.range n, f j := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ, Finset.sum_range_succ, ih]

theorem input_cell (s : State ℝ) (read : Nat → ℝ) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (api : Host.Inputs s read w b x W B X) (h : Contract.InputContract w b x)
    (row col : Nat) (hr : row < Contract.rows x) (hc : col < Contract.width x) :
    (s.mem (region X) (col + Contract.width x * row)).read .real =
      logicalRead read X (col + Contract.width x * row) := by
  have hi : col + Contract.width x * row < x.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel, Contract.input_shape x h.rank] using output_bounds _ _ _ _ hr hc
  rw [coordinate s read x X api.inputAPI api.inputMetadata api.inputMemory h.input _ hi]
  rfl

theorem moments (s : State ℝ) (read : Nat → ℝ) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (api : Host.Inputs s read w b x W B X) (h : Contract.InputContract w b x)
    (row : Nat) (hr : row < Contract.rows x) :
    logicalMean s (region X) (Contract.width x) row =
      Contract.mean (Contract.width x) (fun col => logicalRead read X (col + Contract.width x * row)) ∧
    logicalVariance s (region X) (Contract.width x) row =
      Contract.variance (Contract.width x) (fun col => logicalRead read X (col + Contract.width x * row)) := by
  have hm : logicalMean s (region X) (Contract.width x) row =
      Contract.mean (Contract.width x) (fun col => logicalRead read X (col + Contract.width x * row)) := by
    unfold logicalMean logicalRow Contract.mean
    rw [sum_range]
    congr 1
    apply Finset.sum_congr rfl
    intro j hj
    exact input_cell s read w b x W B X api h row j hr (Finset.mem_range.mp hj)
  refine ⟨hm, ?_⟩
  unfold logicalVariance logicalRow Contract.variance
  rw [List.map_map, sum_range, hm]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Function.comp_apply, input_cell s read w b x W B X api h row j hr (Finset.mem_range.mp hj)]

theorem output (s : State ℝ) (read : Nat → ℝ) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (api : Host.Inputs s read w b x W B X) (h : Contract.InputContract w b x)
    (row col : Nat) (hr : row < Contract.rows x) (hc : col < Contract.width x) :
    logicalOutput s (region X) (region W) (region B) (Contract.width x) row col =
      Contract.output (Contract.width x) (fun j => logicalRead read X (j + Contract.width x * row))
        (logicalRead read W) (logicalRead read B) col := by
  have hw : col < w.numel := by simpa [VeriTile.Torch.TensorMetadata.numel, h.weightShape] using hc
  have hb : col < b.numel := by simpa [VeriTile.Torch.TensorMetadata.numel, h.biasShape] using hc
  unfold logicalOutput logicalInverse Contract.output
  rw [(moments s read w b x W B X api h row hr).1,
    (moments s read w b x W B X api h row hr).2,
    input_cell s read w b x W B X api h row col hr hc,
    coordinate s read w W api.weightAPI api.weightMetadata api.weightMemory h.weight col hw,
    coordinate s read b B api.biasAPI api.biasMetadata api.biasMemory h.bias col hb]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Logical
