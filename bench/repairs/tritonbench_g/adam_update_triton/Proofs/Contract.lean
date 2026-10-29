import bench.repairs.tritonbench_g.adam_update_triton.Contract
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Contract
namespace VeriTile.Bench.Repairs.AdamUpdateTriton
abbrev TensorBinding := L2NormTriton1.TensorBinding

theorem contract_iff (p g m : TensorMetadata) : InputContract p g m ↔
    (p.layout=.strided ∧ g.layout=.strided ∧ m.layout=.strided) ∧
    (p.dtype=.floating .fp32 ∧ g.dtype=.floating .fp32 ∧ m.dtype=.floating .fp32) ∧
    (p.device.kind="cuda" ∧ g.device.kind="cuda" ∧ m.device.kind="cuda") ∧
    (g.device=p.device ∧ m.device=p.device) ∧ (p.numel≤g.numel ∧ p.numel≤m.numel) ∧
    (p.isContiguous=Bool.true ∧ g.isContiguous=Bool.true ∧ m.isContiguous=Bool.true) ∧
    (p.isNeg=Bool.false ∧ g.isNeg=Bool.false ∧ m.isNeg=Bool.false) ∧
    (0<p.numel → p.dataPtr.toNat%16=0 ∧ g.dataPtr.toNat%16=0 ∧ m.dataPtr.toNat%16=0) ∧
    (0<p.numel → DisjointRanges p g ∧ DisjointRanges p m ∧ DisjointRanges g m) := by
  constructor
  · intro h;exact ⟨h.layout,h.dtype,h.cuda,h.sameDevice,h.enough,h.contiguous,h.noNeg,h.alignment,h.separate⟩
  · rintro ⟨hl,ht,hc,hd,he,hs,hn,ha,hx⟩;exact ⟨hl,ht,hc,hd,he,hs,hn,ha,hx⟩

theorem checkTensorInputs_iff (p g m : TensorMetadata) :
    checkTensorInputs p g m=.ok () ↔ InputContract p g m := by
  rw [contract_iff]
  by_cases empty : p.numel=0
  · simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,empty,ite_eq_iff,and_assoc]
  · have positive : 0<p.numel := Nat.pos_of_ne_zero empty
    simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,empty,positive,ite_eq_iff,and_assoc]

theorem checkInputs_iff (p g m : ArgumentMetadata) :
    checkInputs p g m=.ok () ↔ AdmittedArguments p g m := by
  cases p <;> cases g <;> cases m <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AdamUpdateTriton
