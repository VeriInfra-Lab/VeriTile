import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Contract
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Contract
open VeriTile.Torch

theorem contract_iff (scaled : Bool) (a b c s : TensorMetadata) (p : Config) :
    InputContract scaled a b c s p ↔
      Layouts scaled a b c s ∧ DTypes scaled a b c s ∧ CUDA scaled a b c s ∧
      SameDevice scaled a b c s ∧ Ranks scaled a b c s ∧ Shapes scaled a b c s ∧
      Contiguous scaled a b c s ∧ Resolved scaled a b c s ∧ ConfigContract p ∧
      ExecutionContract scaled a b c s p := by
  constructor
  · intro h
    exact ⟨h.layouts,h.dtypes,h.cuda,h.device,h.ranks,h.shapes,h.contiguous,h.resolved,h.config,h.execution⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j⟩
    exact ⟨a,b,c,d,e,f,g,h,i,j⟩

theorem checkTensorInputs_iff (scaled : Bool) (a b c s : TensorMetadata) (p : Config) :
    checkTensorInputs scaled a b c s p = .ok () ↔ InputContract scaled a b c s p := by
  rw [contract_iff]
  by_cases empty : Empty a b <;>
    simp [checkTensorInputs,ExecutionContract,empty,ite_eq_iff,and_assoc]

theorem checkInputs_iff (scaled : Bool) (a b c s : ArgumentMetadata) (p : ConfigArgument) :
    checkInputs scaled a b c s p = .ok () ↔ AdmittedArguments scaled a b c s p := by
  cases scaled <;> cases a <;> cases b <;> cases c <;> cases s <;> cases p <;>
    simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem matrix_shape (t : TensorMetadata) (h : t.shape.length = 2) :
    t.shape = [rows t,columns t] := by
  rcases ht : t.shape with _ | ⟨a, _ | ⟨b,rest⟩⟩ <;> simp [ht] at h
  have hn : rest = [] := by simpa using h
  subst rest
  simp [rows,columns,ht]

theorem matrix_numel (t : TensorMetadata) (h : t.shape.length = 2) :
    t.numel = rows t*columns t := by
  simp [TensorMetadata.numel,matrix_shape t h]

theorem execution_of_nonempty (scaled : Bool) (a b c s : TensorMetadata) (p : Config)
    (h : InputContract scaled a b c s p) (nonempty : ¬Empty a b) :
    IndexEnvelope (rows a) (columns b) (columns a) p ∧
      Windows scaled a b c s ∧ Nonoverlap scaled a b c s := h.execution.resolve_left nonempty

theorem positive_tiles (p : Config) (h : ConfigContract p) :
    0 < p.blockM ∧ 0 < p.blockN ∧ 0 < p.blockK ∧ 0 < p.groupM := by
  have := h.1
  exact ⟨by omega,by omega,by omega,h.2.2.2.1⟩

theorem envelope_fields (M N K : Nat) (p : Config) (h : IndexEnvelope M N K p) :
    M+p.blockM-1 < 2^31 ∧ N+p.blockN-1 < 2^31 ∧ K+p.blockK-1 < 2^31 ∧
    p.groupM*ceil N p.blockN < 2^31 ∧ ceil M p.blockM*ceil N p.blockN < 2^31 ∧
    padded M p.blockM*K+padded K p.blockK+p.blockK < 2^31 ∧
    (padded K p.blockK+p.blockK)*N+padded N p.blockN < 2^31 ∧
    padded M p.blockM*N+padded N p.blockN < 2^31 := by
  simpa [IndexEnvelope,indexExpressions] using h

#axiomsClean checkInputs_iff
#axiomsClean execution_of_nonempty
#axiomsClean envelope_fields
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Contract
