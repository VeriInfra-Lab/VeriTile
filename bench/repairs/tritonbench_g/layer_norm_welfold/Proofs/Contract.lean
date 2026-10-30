import bench.repairs.tritonbench_g.layer_norm_welfold.Contract
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeABI

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Admission
open VeriTile.Triton FP.Structural Reuse

theorem contract_iff (W Bias X : Metadata) (S D : Nat) : Contract W Bias X S D ↔
    (W.tensor.layout = .strided ∧ Bias.tensor.layout = .strided ∧ X.tensor.layout = .strided) ∧
    (W.tensor.dtype = .floating .fp32 ∧ Bias.tensor.dtype = .floating .fp32 ∧ X.tensor.dtype = .floating .fp32) ∧
    (W.tensor.device = ⟨"cuda", 0⟩ ∧ Bias.tensor.device = ⟨"cuda", 0⟩ ∧ X.tensor.device = ⟨"cuda", 0⟩) ∧
    X.tensor.shape = [S,D] ∧ 0 < D ∧
    (W.tensor.shape = [D] ∧ Bias.tensor.shape = [D]) ∧
    (X.strides = [D,1] ∧ W.strides = [1] ∧ Bias.strides = [1]) ∧
    (W.storageOffset = 0 ∧ Bias.storageOffset = 0 ∧ X.storageOffset = 0) ∧
    (W.tensor.isNeg = Bool.false ∧ Bias.tensor.isNeg = Bool.false ∧ X.tensor.isNeg = Bool.false) ∧
    (D + 2048 < 2^31 ∧ S * D + 2048 < 2^31) ∧
    (0 < S * D → X.tensor.dataPtr.toNat % 16 = 0) ∧
    W.tensor.dataPtr.toNat % 16 = 0 ∧ Bias.tensor.dataPtr.toNat % 16 = 0 := by
  constructor
  · intro h
    exact ⟨h.layout, h.dtype, h.device, h.xShape, h.width, h.affineShape, h.strides,
      h.offset, h.noNeg, h.extent, h.alignment⟩
  · rintro ⟨hl,hd,hv,hx,hw,ha,hs,ho,hn,he,hal⟩
    exact ⟨hl,hd,hv,hx,hw,ha,hs,ho,hn,he,hal⟩

theorem checkTensors_iff (W Bias X : Metadata) (S D : Nat) :
    checkTensors W Bias X = .ok (S,D) ↔ Contract W Bias X S D := by
  rw [contract_iff]
  rcases shape : X.tensor.shape with _ | ⟨s,rest⟩
  · simp [checkTensors, shape, ite_eq_iff]
  rcases rest with _ | ⟨d,rest⟩
  · simp [checkTensors, shape, ite_eq_iff]
  rcases rest with _ | ⟨extra,rest⟩
  · simp [checkTensors, checkDimensions, shape, ite_eq_iff, and_assoc,
      Nat.pos_iff_ne_zero]
    aesop <;> omega
  · simp [checkTensors, shape, ite_eq_iff]

theorem checkInputs_iff (W Bias X : Argument) (S D : Nat) :
    checkInputs W Bias X = .ok (S,D) ↔ Admitted W Bias X S D := by
  cases W <;> cases Bias <;> cases X <;>
    simp [checkInputs, Admitted, checkTensors_iff]

/-- Actual Tensor observations attach checked metadata to the source object
and backing region. Typed cells are ordinary FP32 storage interpretation. -/
structure TensorBinding {α : Type} (s : State α) (m : Metadata) (t : Tensor) : Prop where
  shape : t.shape = m.tensor.shape
  strides : t.strides = m.strides
  dtype : m.tensor.dtype = .floating .fp32 → t.dtype = .fp32
  offset : t.storageOffset = m.storageOffset
  device : t.device = m.tensor.device.index
  cells : ∀ i, i < m.tensor.numel → ∃ value, s.mem t.region i = .mk .real value

theorem source_domain {α : Type} (s : State α) (mw mb mx : Metadata)
    (W Bias X : Tensor) (S D : Nat) (h : Contract mw mb mx S D)
    (bw : TensorBinding s mw W) (bb : TensorBinding s mb Bias) (bx : TensorBinding s mx X) :
    Reuse.InputContract s X W Bias S D ∧ Native.IndexABI S D := by
  refine ⟨⟨h.width, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨h.width, h.extent⟩⟩
  · exact ⟨bx.shape.trans h.xShape, bx.strides.trans h.strides.1, bx.dtype h.dtype.2.2,
      bx.offset.trans h.offset.2.2, bx.device.trans (congrArg VeriTile.Torch.Device.index h.device.2.2)⟩
  · exact ⟨bw.shape.trans h.affineShape.1, bw.strides.trans h.strides.2.1, bw.dtype h.dtype.1,
      bw.offset.trans h.offset.1, bw.device.trans (congrArg VeriTile.Torch.Device.index h.device.1)⟩
  · exact ⟨bb.shape.trans h.affineShape.2, bb.strides.trans h.strides.2.2, bb.dtype h.dtype.2.1,
      bb.offset.trans h.offset.2.1, bb.device.trans (congrArg VeriTile.Torch.Device.index h.device.2.1)⟩
  · intro row hr col hc
    apply bx.cells
    simp only [VeriTile.Torch.TensorMetadata.numel, h.xShape, List.prod_cons, List.prod_nil, Nat.mul_one]
    nlinarith
  · intro col hc
    apply bw.cells
    simpa only [VeriTile.Torch.TensorMetadata.numel, h.affineShape.1, List.prod_cons, List.prod_nil, Nat.mul_one] using hc
  · intro col hc
    apply bb.cells
    simpa only [VeriTile.Torch.TensorMetadata.numel, h.affineShape.2, List.prod_cons, List.prod_nil, Nat.mul_one] using hc

#axiomsClean checkInputs_iff
#axiomsClean source_domain
end VeriTile.Bench.Repairs.LayerNormWelfold.Admission
