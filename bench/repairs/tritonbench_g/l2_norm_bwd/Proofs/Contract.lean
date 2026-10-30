import bench.repairs.tritonbench_g.l2_norm_bwd.Contract
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Contract

namespace VeriTile.Bench.Repairs.L2NormBwd
abbrev TensorBinding := L2NormTriton1.TensorBinding

theorem contract_iff (x dy : TensorMetadata) : InputContract x dy ↔
    (x.layout=.strided ∧ dy.layout=.strided) ∧
    (x.dtype=.floating .fp32 ∧ dy.dtype=.floating .fp32) ∧
    (x.device.kind="cuda" ∧ dy.device.kind="cuda") ∧ x.device=dy.device ∧
    dy.shape≠[] ∧ (0<dy.shape.getLastD 0 ∧ dy.shape.getLastD 0≤16384) ∧
    x.numel%dy.shape.getLastD 0=0 ∧ x.numel≤dy.numel ∧
    (x.isContiguous=Bool.true ∧ dy.isContiguous=Bool.true) ∧
    (x.isNeg=Bool.false ∧ dy.isNeg=Bool.false) ∧
    (0<x.numel → x.dataPtr.toNat%4=0 ∧ dy.dataPtr.toNat%4=0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.sameDevice,h.rank,h.width,h.divisible,h.enough,h.contiguous,h.noNeg,h.alignment⟩
  · rintro ⟨hl,hd,hc,hv,hr,hw,he,hs,ht,hn,ha⟩; exact ⟨hl,hd,hc,hv,hr,hw,he,hs,ht,hn,ha⟩

theorem checkTensorInputs_iff (x dy : TensorMetadata) :
    checkTensorInputs x dy=.ok () ↔ InputContract x dy := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x dy : ArgumentMetadata) :
    checkInputs x dy=.ok () ↔ AdmittedArguments x dy := by
  cases x <;> cases dy <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.L2NormBwd
