import bench.repairs.tritonbench_g.matrix_vector_multip.Contract
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip
open VeriTile.Torch

theorem contract_iff (x y : TensorMetadata) : InputContract x y ↔
    (x.layout = .strided ∧ y.layout = .strided) ∧
    (x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32) ∧
    (x.device.kind = "cuda" ∧ y.device.kind = "cuda") ∧
    x.device = y.device ∧ (x.shape.length = 2 ∧ y.shape.length = 1) ∧
    columns x = vectorLength y ∧
    (x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true) ∧
    (x.isNeg = Bool.false ∧ y.isNeg = Bool.false) ∧
    (rows x < 2^63 ∧ columns x < 2^63) ∧
    ((0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧
     (0 < y.numel → y.dataPtr.toNat % 4 = 0)) := by
  constructor
  · intro h
    exact ⟨h.layout,h.dtype,h.cuda,h.device,h.rank,h.dimensions,h.contiguous,h.noNeg,h.extent,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j⟩
    exact ⟨a,b,c,d,e,f,g,h,i,j⟩

theorem checkTensorInputs_iff (x y : TensorMetadata) :
    checkTensorInputs x y = .ok () ↔ InputContract x y := by
  rw [contract_iff]
  simp [checkTensorInputs, TensorMetadata.isCuda, ite_eq_iff]

theorem checkInputs_iff (x y : ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := by
  cases x <;> cases y <;> simp [checkInputs, AdmittedArguments, checkTensorInputs_iff]

theorem matrix_shape (x y : TensorMetadata) (h : InputContract x y) :
    x.shape = [rows x, columns x] := by
  have hr := h.rank.1
  rcases hx : x.shape with _ | ⟨a, _ | ⟨b,rest⟩⟩ <;> simp [hx] at hr
  have hn : rest = [] := by simpa using hr
  subst rest
  simp [rows, columns, hx]

theorem vector_shape (x y : TensorMetadata) (h : InputContract x y) :
    y.shape = [columns x] := by
  have hr := h.rank.2
  rcases hy : y.shape with _ | ⟨a,rest⟩ <;> simp [hy] at hr
  have hn : rest = [] := by simpa using hr
  subst rest
  have hc := h.dimensions
  simp [vectorLength, hy] at hc
  simp [hc]

theorem matrix_numel (x y : TensorMetadata) (h : InputContract x y) :
    x.numel = rows x * columns x := by
  simp [TensorMetadata.numel, matrix_shape x y h]

theorem vector_numel (x y : TensorMetadata) (h : InputContract x y) :
    y.numel = columns x := by
  simp [TensorMetadata.numel, vector_shape x y h]

#axiomsClean checkInputs_iff
#axiomsClean matrix_shape
#axiomsClean vector_shape
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatrixVectorMultip
