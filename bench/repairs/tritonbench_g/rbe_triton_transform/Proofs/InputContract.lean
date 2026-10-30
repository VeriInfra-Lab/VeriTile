import bench.repairs.tritonbench_g.rbe_triton_transform.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Contract
open VeriTile.Torch

theorem checkInputs_iff (x : TensorMetadata) (pos : Position) :
    checkInputs x pos=.ok () ↔ InputContract x pos := by
  have unpack : InputContract x pos ↔ x.layout=.strided ∧ x.dtype=.floating .fp32 ∧
      x.device.kind="cuda" ∧ x.shape.length=3 ∧ x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧
      (0<x.numel → x.dataPtr.toNat%4=0) ∧ pos.integer=Bool.true ∧
      -(2^63 : Int)≤pos.value ∧ pos.value<(2^63 : Int) ∧
      (0<x.numel → pos.value+(paddedRows x : Int)-1<(2^63 : Int)) :=
    ⟨fun h => ⟨h.layout,h.dtype,h.cuda,h.rank,h.contiguous,h.resolved,h.aligned,h.integer,h.lower,h.upper,h.padded⟩,
      fun ⟨a,b,c,d,e,f,g,h,i,j,k⟩ => ⟨a,b,c,d,e,f,g,h,i,j,k⟩⟩
  rw [unpack]
  simp [checkInputs,ite_eq_iff,and_assoc]

theorem checkArguments_iff (x : ArgumentMetadata) (pos : Position) :
    checkArguments x pos=.ok () ↔ AdmittedArguments x pos := by
  cases x <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RbeTritonTransform.Contract
