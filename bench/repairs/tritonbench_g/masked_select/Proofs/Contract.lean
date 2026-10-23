import bench.repairs.tritonbench_g.masked_select.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MaskedSelect.Whole (broadcastShape)

theorem input_contract_iff (x m : TensorMetadata) (shape : List Nat) :
    InputContract x m shape ↔
      (x.layout = .strided ∧ m.layout = .strided) ∧
      (x.dtype = .floating .fp32 ∧ m.dtype = .other "bool") ∧
      (x.device.kind = "cuda" ∧ m.device.kind = "cuda") ∧
      x.device = m.device ∧ m.isNeg = Bool.false ∧
      broadcastShape x.shape m.shape = some shape ∧
      (0 < shape.prod ∧ shape.prod < 2^31) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.device,h.maskResolved,h.broadcast,h.extent⟩
  · rintro ⟨hl,hd,hc,hv,hn,hb,he⟩; exact ⟨hl,hd,hc,hv,hn,hb,he⟩

theorem checkTensorInputs_iff (x m : TensorMetadata) (shape : List Nat) :
    checkTensorInputs x m = .ok shape ↔ InputContract x m shape := by
  rw [input_contract_iff]
  cases hb : broadcastShape x.shape m.shape with
  | none => simp [checkTensorInputs,hb,TensorMetadata.isCuda,ite_eq_iff]
  | some s =>
    by_cases hs : s = shape
    · subst s
      simp [checkTensorInputs,hb,TensorMetadata.isCuda,ite_eq_iff,Nat.pos_iff_ne_zero]
      aesop
    · simp [checkTensorInputs,hb,hs,TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x m : ArgumentMetadata) (shape : List Nat) :
    checkInputs x m = .ok shape ↔ AdmittedArguments x m shape := by
  cases x <;> cases m <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem checkPrepared_iff (pointer : Nat) :
    checkPrepared pointer = .ok () ↔ PreparedContract pointer := by
  simp [checkPrepared,PreparedContract]

#axiomsClean checkTensorInputs_iff
#axiomsClean checkInputs_iff
#axiomsClean checkPrepared_iff
end VeriTile.Bench.Repairs.MaskedSelect
