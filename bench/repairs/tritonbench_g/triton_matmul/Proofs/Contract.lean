import bench.repairs.tritonbench_g.triton_matmul.Contract
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Contract
open VeriTile.Torch

theorem contract_iff (a b : TensorMetadata) : InputContract a b ↔
    (a.layout=.strided ∧ b.layout=.strided) ∧ (Supported a ∧ b.dtype=a.dtype) ∧
    (a.device.kind="cuda" ∧ b.device.kind="cuda" ∧ b.device=a.device) ∧
    (a.shape.length=2 ∧ b.shape.length=2) ∧ columns a=rows b ∧
    (a.isContiguous=Bool.true ∧ b.isContiguous=Bool.true) ∧ (a.isNeg=Bool.false ∧ b.isNeg=Bool.false) ∧
    Dimensions a b ∧ Execution a b := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.device,h.rank,h.inner,h.contiguous,h.resolved,h.dimensions,h.execution⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩; exact ⟨a,b,c,d,e,f,g,h,i⟩

theorem checkTensorInputs_iff (a b : TensorMetadata) : checkTensorInputs a b=.ok () ↔ InputContract a b := by
  rw [contract_iff]
  by_cases empty : Empty a b <;> simp [checkTensorInputs,Execution,empty,ite_eq_iff,and_assoc]

theorem checkInputs_iff (a b : ArgumentMetadata) : checkInputs a b=.ok () ↔ AdmittedArguments a b := by
  cases a <;> cases b <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem matrix_shape (t : TensorMetadata) (h : t.shape.length=2) : t.shape=[rows t,columns t] := by
  rcases ht : t.shape with _ | ⟨a,_ | ⟨b,rest⟩⟩ <;> simp [ht] at h
  have hn : rest=[] := by simpa using h
  subst rest
  simp [rows,columns,ht]

theorem matrix_numel (t : TensorMetadata) (h : t.shape.length=2) : t.numel=rows t*columns t := by
  simp [TensorMetadata.numel,matrix_shape t h]

theorem execution_nonempty (a b : TensorMetadata) (h : InputContract a b) (nonempty : ¬Empty a b) :
    Envelope a b ∧ Window a ∧ Window b := h.execution.resolve_left nonempty

#axiomsClean checkInputs_iff
#axiomsClean execution_nonempty
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Contract
