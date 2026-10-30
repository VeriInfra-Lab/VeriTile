import bench.tritonbench_g.matmul_triton1.Proofs.Contract
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract
open VeriTile.Torch

theorem contract_iff (fixed default : Bool) (x y : TensorMetadata) : InputContract fixed default x y ↔
    (x.layout = .strided ∧ y.layout = .strided) ∧
    (x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32) ∧
    (x.device.kind = "cuda" ∧ y.device.kind = "cuda") ∧
    x.device = y.device ∧ (fixed = Bool.true → x.device.index = 0) ∧
    (fixed = Bool.true → default = Bool.true) ∧
    (x.shape.length = 2 ∧ y.shape.length = 2) ∧ columns x = rows y ∧
    (x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true) ∧
    (x.isNeg = Bool.false ∧ y.isNeg = Bool.false) ∧
    (rows x < 2^63 ∧ columns x < 2^63 ∧ columns y < 2^63) ∧
    ((0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧ (0 < y.numel → y.dataPtr.toNat % 4 = 0)) := by
  constructor
  · intro h
    exact ⟨h.layout,h.dtype,h.cuda,h.device,h.fixedDevice,h.defaultDtype,h.rank,
      h.dimensions,h.contiguous,h.noNeg,h.extent,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩
    exact ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩

theorem checkTensorInputs_iff (fixed default : Bool) (x y : TensorMetadata) :
    checkTensorInputs fixed default x y = .ok () ↔ InputContract fixed default x y := by
  rw [contract_iff]
  cases fixed <;> cases default <;> simp [checkTensorInputs,TensorMetadata.isCuda,ite_eq_iff,and_assoc]

theorem checkInputs_iff (fixed default : Bool) (x y : ArgumentMetadata) :
    checkInputs fixed default x y = .ok () ↔ AdmittedArguments fixed default x y := by
  cases x <;> cases y <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem matrix_shape (x : TensorMetadata) (h : x.shape.length = 2) :
    x.shape = [rows x,columns x] := by
  rcases hx : x.shape with _ | ⟨a, _ | ⟨b,rest⟩⟩ <;> simp [hx] at h
  have hn : rest = [] := by simpa using h
  subst rest
  simp [rows,columns,hx]

theorem matrix_numel (x : TensorMetadata) (h : x.shape.length = 2) :
    x.numel = rows x * columns x := by
  simp [TensorMetadata.numel,matrix_shape x h]

#axiomsClean checkInputs_iff
#axiomsClean matrix_numel
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract
