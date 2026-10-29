import bench.repairs.tritonbench_g.layer_norm_ops.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LayerNormOps.Admission

theorem tensorCheck_iff (arg : Argument) (shape strides : List Nat) :
    tensorCheck arg shape strides = Bool.true ↔ TensorContract arg shape strides := by
  cases arg <;> simp [tensorCheck, TensorContract, VeriTile.Torch.TensorMetadata.isStrided,
    VeriTile.Torch.TensorMetadata.isFP32, and_assoc, Nat.pos_iff_ne_zero, imp_iff_not_or]

theorem optionalCheck_iff (arg : Argument) (shape strides : List Nat) :
    optionalCheck arg shape strides = Bool.true ↔ OptionalContract arg shape strides := by
  cases arg <;> simp [optionalCheck, OptionalContract, tensorCheck_iff]

theorem scalarCheck_iff (arg : ScalarArgument) : scalarCheck arg = Bool.true ↔ ScalarContract arg := by
  classical
  cases arg with
  | other => simp [scalarCheck, ScalarContract]
  | «float» value => cases value <;> simp [scalarCheck, ScalarContract]

theorem commonCheck_iff (c : Common) : commonCheck c = Bool.true ↔ CommonContract c := by
  simp [commonCheck, CommonContract, tensorCheck_iff, optionalCheck_iff, scalarCheck_iff, and_assoc]

theorem forwardCheck_iff (c : ForwardCall) : forwardCheck c = Bool.true ↔ ForwardContract c := by
  simp [forwardCheck, ForwardContract, commonCheck_iff, optionalCheck_iff, and_assoc]
  cases c.residual.present <;> simp

theorem meanCheck_iff (c : BackwardCall) : meanCheck c = Bool.true ↔ MeanContract c := by
  cases h : c.common.rms.getD Bool.false <;> simp [meanCheck, MeanContract, h, tensorCheck_iff]

theorem backwardCheck_iff (c : BackwardCall) : backwardCheck c = Bool.true ↔ BackwardContract c := by
  simp [backwardCheck, BackwardContract, commonCheck_iff, tensorCheck_iff, optionalCheck_iff, meanCheck_iff, and_assoc]

theorem blockSize_properties (N : Nat) (_positive : 0 < N) (limit : N ≤ 16384) :
    N ≤ blockSize N ∧ blockSize N ≤ 16384 ∧ ∃ k, blockSize N = 2^k := by
  have expBound : Nat.clog 2 N ≤ 14 := Nat.clog_le_of_le_pow (by norm_num; exact limit)
  have bound : 2 ^ Nat.clog 2 N ≤ 16384 := by
    have := Nat.pow_le_pow_right (by decide : 0 < (2 : Nat)) expBound
    norm_num at this ⊢
    exact this
  rw [blockSize, Nat.min_eq_right bound]
  exact ⟨Nat.le_pow_clog (by decide) N, bound, ⟨_, rfl⟩⟩

#axiomsClean forwardCheck_iff
#axiomsClean backwardCheck_iff
#axiomsClean blockSize_properties
end VeriTile.Bench.Repairs.LayerNormOps.Admission
