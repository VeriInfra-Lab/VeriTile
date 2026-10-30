import bench.tritonbench_g.layer_norm_liger.Proofs.Preparation

namespace VeriTile.Bench.Repairs.LayerNormLiger.LogicalInputs
open VeriTile.Triton FP.Structural VeriTile.Torch
open Logical Preparation
noncomputable section

theorem mean_congr (n : Nat) (x z : Nat → ℝ) (h : ∀ i, i < n → x i = z i) : mean n x = mean n z := by
  unfold mean sum
  congr 1
  exact Finset.sum_congr rfl (fun i hi => h i (Finset.mem_range.mp hi))

theorem rstd_congr (n : Nat) (eps : ℝ) (x z : Nat → ℝ) (h : ∀ i, i < n → x i = z i) :
    rstd n eps x = rstd n eps z := by
  have mean := mean_congr n x z h
  have variance : variance n x = variance n z := by
    unfold Logical.variance sum
    rw [mean]
    congr 1
    exact Finset.sum_congr rfl (fun i hi => by dsimp only; rw [h i (Finset.mem_range.mp hi)])
  rw [rstd,rstd,variance]

theorem y_congr (n i : Nat) (eps : ℝ) (x w b z v c : Nat → ℝ) (hi : i < n)
    (hx : ∀ j, j < n → x j = z j) (hw : ∀ j, j < n → w j = v j) (hb : ∀ j, j < n → b j = c j) :
    y n eps x w b i = y n eps z v c i := by
  unfold y
  rw [hx i hi,hw i hi,hb i hi,mean_congr n x z hx,rstd_congr n eps x z hx]

theorem row (s : State ℝ) (x w b dy : Tensor ℝ) (p : Prepared ℝ) (a : Wrapper.Allocations)
    (eps : Contract.Epsilon) (h : Contract.InputContract x.metadata w.metadata b.metadata eps)
    (prepared : Result x w b dy p) (ready : Ready s p) (row col : Nat)
    (hr : row < Contract.rows x.metadata) (hc : col < Contract.width x.metadata) :
    rowX (Wrapper.forward (regions p) a x.metadata eps) (atPid s row) col =
      x.logical (row * Contract.width x.metadata + col) := by
  have hi : row * Contract.width x.metadata + col < x.metadata.numel := by
    rw [Contract.shape_elements x.metadata h.rank]
    have bound := Nat.mul_le_mul_right (Contract.width x.metadata) hr
    nlinarith
  have cell := input_cells s x w b dy p prepared ready h.rank _ hi
  simpa [rowX,Wrapper.forward,atPid,cell] using congrArg (fun c => c.read .fp32) cell

theorem weight (s : State ℝ) (x w b dy : Tensor ℝ) (p : Prepared ℝ) (a : Wrapper.Allocations)
    (eps : Contract.Epsilon) (h : Contract.InputContract x.metadata w.metadata b.metadata eps)
    (prepared : Result x w b dy p) (ready : Ready s p) (col : Nat) (hc : col < Contract.width x.metadata) :
    Logical.weight (Wrapper.forward (regions p) a x.metadata eps) s col = w.logical col := by
  have hw : col < w.metadata.numel := by simpa [TensorMetadata.numel,h.weightShape] using hc
  have size : p.weight.metadata.numel = w.metadata.numel := by simp [TensorMetadata.numel,prepared.weightShape]
  have cell := ready.weight col (by simpa [size] using hw)
  rw [prepared.weight col hw] at cell
  exact congrArg (fun c => c.read .fp32) cell

theorem bias (s : State ℝ) (x w b dy : Tensor ℝ) (p : Prepared ℝ) (a : Wrapper.Allocations)
    (eps : Contract.Epsilon) (h : Contract.InputContract x.metadata w.metadata b.metadata eps)
    (prepared : Result x w b dy p) (ready : Ready s p) (col : Nat) (hc : col < Contract.width x.metadata) :
    Logical.bias (Wrapper.forward (regions p) a x.metadata eps) s col = b.logical col := by
  have hb : col < b.metadata.numel := by simpa [TensorMetadata.numel,h.biasShape] using hc
  have size : p.bias.metadata.numel = b.metadata.numel := by simp [TensorMetadata.numel,prepared.biasShape]
  have cell := ready.bias col (by simpa [size] using hb)
  rw [prepared.bias col hb] at cell
  exact congrArg (fun c => c.read .fp32) cell

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.LogicalInputs
