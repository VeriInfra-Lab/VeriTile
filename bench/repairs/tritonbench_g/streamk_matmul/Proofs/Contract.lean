import bench.repairs.tritonbench_g.streamk_matmul.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.StreamK.Checked.Contract
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.StreamK
open Whole001

theorem contract_iff (a b : TensorMetadata) (c : Config) : InputContract a b c ↔
    TensorInputs a b ∧ Profile c ∧ (Empty a b ∨ (Control (plan a b c) ∧ StreamEnvelope (plan a b c) ∧
    ByteEnvelope (plan a b c) ∧ Window a ∧ Window b)) := by
  constructor
  · intro h; exact ⟨h.inputs,h.profile,h.execution⟩
  · rintro ⟨i,p,e⟩; exact ⟨i,p,e⟩

theorem checkTensorInputs_iff (a b : TensorMetadata) (c : Config) :
    checkTensorInputs a b c=.ok () ↔ InputContract a b c := by
  rw [contract_iff]
  by_cases empty : Empty a b <;> simp [checkTensorInputs,empty,ite_eq_iff,and_assoc]

theorem checkInputs_iff (a b : ArgumentMetadata) (c : Config) :
    checkInputs a b c=.ok () ↔ AdmittedArguments a b c := by
  cases a <;> cases b <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem matrix_shape (t : TensorMetadata) (h : t.shape.length=2) : t.shape=[rows t,columns t] := by
  rcases ht : t.shape with _ | ⟨a,_ | ⟨b,rest⟩⟩ <;> simp [ht] at h
  have hn : rest=[] := by simpa using h
  subst rest
  simp [rows,columns,ht]

theorem matrix_numel (t : TensorMetadata) (h : t.shape.length=2) : t.numel=rows t*columns t := by
  simp [TensorMetadata.numel,matrix_shape t h]

theorem execution_nonempty (a b : TensorMetadata) (c : Config) (h : InputContract a b c) (nonempty : ¬Empty a b) :
    Control (plan a b c) ∧ StreamEnvelope (plan a b c) ∧ ByteEnvelope (plan a b c) ∧ Window a ∧ Window b :=
  h.execution.resolve_left nonempty

#axiomsClean checkInputs_iff
#axiomsClean execution_nonempty
end VeriTile.Bench.Repairs.StreamK.Checked.Contract
